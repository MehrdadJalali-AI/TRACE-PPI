"""Lightweight semantic embeddings (transformer optional, TF-IDF+SVD fallback)."""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def embed_profiles(
    profiles: pd.DataFrame,
    n_components: int = 64,
) -> Tuple[np.ndarray, str, List[str]]:
    texts = profiles["text_profile"].fillna("").astype(str).tolist()
    protein_ids = profiles["protein_id"].tolist()
    mode = "fallback_tfidf_svd"
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(texts, show_progress_bar=False)
        emb = normalize(emb)
        mode = "sentence_transformers_all-MiniLM-L6-v2"
        return emb.astype(np.float32), mode, protein_ids
    except Exception as e:
        logger.info("Transformer unavailable (%s); using TF-IDF+SVD", e)
        # TODO: re-run with Sentence-BERT and report the delta in the Gavin benchmark table.

    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    dim = min(n_components, max(2, X.shape[1] - 1), X.shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    emb = normalize(svd.fit_transform(X))
    return emb.astype(np.float32), mode, protein_ids
