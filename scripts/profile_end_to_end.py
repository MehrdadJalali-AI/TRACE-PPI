#!/usr/bin/env python3
"""End-to-end wall clock and peak RSS for the reported TRACE-PPI path.

Each dataset and embedding mode runs in a fresh process, three times.
Sentence-BERT is blocked so a rebuild stays on TF-IDF plus SVD.
On macOS, ru_maxrss is bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "end_to_end_profile.json"


class _BlockSentenceTransformers:
    def find_spec(self, fullname, path, target=None):
        if fullname == "sentence_transformers" or fullname.startswith("sentence_transformers."):
            raise ImportError("sentence_transformers blocked so timings stay on TF-IDF+SVD")
        return None


def _block_transformers() -> None:
    sys.meta_path.insert(0, _BlockSentenceTransformers())


def _one(dataset: str, embeddings: str) -> dict:
    _block_transformers()
    sys.path.insert(0, str(ROOT))
    import resource
    import yaml

    from trace_ppi.cosmos_v2_runner import run_v2
    from trace_ppi.paths import RESULTS

    cfg = yaml.safe_load((ROOT / "configs" / "trace_ppi_final.yaml").read_text())
    params = {
        "base_method": "mcl_overlap",
        "skip_expansion": True,
        "preserve_all": True,
        "recall_safe_supplement": True,
        "apply_orbit_labels": True,
        **cfg.get("supplement", {}),
    }
    cache = [
        RESULTS / "embeddings" / f"protein_embeddings_{dataset}.npz",
        RESULTS / "embeddings" / f"protein_embedding_index_{dataset}.csv",
        RESULTS / "embeddings" / f"protein_embedding_meta_{dataset}.json",
    ]
    backup = Path(tempfile.mkdtemp(prefix=f"trace_emb_{dataset}_"))
    moved = []
    try:
        if embeddings == "rebuild":
            for src in cache:
                if src.exists():
                    dest = backup / src.name
                    shutil.move(src, dest)
                    moved.append((dest, src))
        t0 = time.perf_counter()
        out = run_v2(dataset, variant="recall_safe_orbits", params=params, write_outputs=False)
        elapsed = time.perf_counter() - t0
    finally:
        for src in cache:
            if src.exists() and embeddings == "rebuild":
                src.unlink()
        for dest, src in moved:
            shutil.move(dest, src)
        shutil.rmtree(backup, ignore_errors=True)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_bytes = int(rss)
    else:
        rss_bytes = int(rss) * 1024
    mode = str(out.get("embedding_mode", ""))
    if "sentence_transformers" in mode:
        raise SystemExit(f"transformer backend leaked into the timing run: {mode}")
    refined = out["refined"]
    sizes = refined.groupby("community_id")["protein_id"].transform("size")
    n_assign = int((sizes >= 2).sum())
    return {
        "dataset": dataset,
        "embeddings": embeddings,
        "seconds": elapsed,
        "rss_bytes": rss_bytes,
        "embedding_mode": mode,
        "assignments_size_ge2": n_assign,
        "supplement_stats": out.get("supplement_stats", {}),
    }


def _summarise(rows: list[dict]) -> dict:
    seconds = [r["seconds"] for r in rows]
    rss = [r["rss_bytes"] / (1024 ** 2) for r in rows]
    return {
        "seconds_mean": statistics.mean(seconds),
        "seconds_sd": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
        "rss_mib_mean": statistics.mean(rss),
        "rss_mib_sd": statistics.stdev(rss) if len(rss) > 1 else 0.0,
        "embedding_modes": sorted({r["embedding_mode"] for r in rows}),
        "assignments_size_ge2": sorted({r["assignments_size_ge2"] for r in rows}),
        "supplement_stats": rows[0].get("supplement_stats", {}),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--dataset", choices=["gavin", "krogan"])
    parser.add_argument("--embeddings", choices=["cached", "rebuild"])
    args = parser.parse_args()
    if args.child:
        print(json.dumps(_one(args.dataset, args.embeddings)))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    summary = {}
    for dataset in ("gavin", "krogan"):
        for embeddings in ("cached", "rebuild"):
            rows = []
            for _ in range(3):
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        "--dataset",
                        dataset,
                        "--embeddings",
                        embeddings,
                    ],
                    cwd=str(ROOT),
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                line = proc.stdout.strip().splitlines()[-1]
                rows.append(json.loads(line))
                print(dataset, embeddings, f"{rows[-1]['seconds']:.2f}s", rows[-1]["rss_bytes"], flush=True)
            summary[f"{dataset}_{embeddings}"] = _summarise(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print("wrote", OUT)
    for key, block in summary.items():
        print(
            key,
            f"time {block['seconds_mean']:.2f}±{block['seconds_sd']:.2f}s",
            f"rss {block['rss_mib_mean']:.1f}±{block['rss_mib_sd']:.1f} MiB",
            "n",
            block["assignments_size_ge2"],
            "mode",
            block["embedding_modes"],
        )


if __name__ == "__main__":
    main()
