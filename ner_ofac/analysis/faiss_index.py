"""FAISS index builder and searcher for semantic OFAC name matching.

Builds per-category ``faiss.IndexFlatL1`` (Manhattan / L1 distance) indexes
from lists of OFAC entity names encoded with a sentence-transformers model.
At search time, a query embedding is compared against the index to retrieve
the *k* nearest neighbours along with their L1 distances.

Lower L1 distance = more similar.  Callers are responsible for converting
distances to similarity scores appropriate for composite scoring.
"""

import logging
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def build_index(
    name_list: List[str],
    model: SentenceTransformer,
) -> Tuple[faiss.IndexFlatL1, List[str]]:
    """Encode *name_list* with *model* and build a FAISS L1 flat index.

    Args:
        name_list: List of entity name strings to index.  Should all belong
            to the same OFAC category (persons OR orgs).
        model: A loaded ``sentence_transformers.SentenceTransformer`` instance
            used to produce fixed-length dense embeddings.

    Returns:
        A 2-tuple of:

        - **index** (``faiss.IndexFlatL1``): The populated FAISS index.
          The number of vectors equals ``len(name_list)``.
        - **name_list** (``List[str]``): The original name list, returned so
          that callers can maintain an aligned mapping from index position →
          name string.

    Raises:
        ValueError: If *name_list* is empty.

    Example::

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index, names = build_index(["JOHN DOE", "JANE DOE"], model)
    """
    if not name_list:
        raise ValueError("name_list must not be empty when building a FAISS index.")

    logger.info("Encoding %d names for FAISS index", len(name_list))
    embeddings = model.encode(name_list, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL1(dimension)
    index.add(embeddings)
    logger.info("FAISS IndexFlatL1 built with %d vectors (dim=%d)", index.ntotal, dimension)
    return index, name_list


def search_index(
    index: faiss.IndexFlatL1,
    name_list: List[str],
    query_embedding: np.ndarray,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Search a FAISS L1 index for the *k* nearest neighbours to *query_embedding*.

    Args:
        index: A populated ``faiss.IndexFlatL1`` index as returned by
            ``build_index``.
        name_list: The list of names that was used to build *index*.  Must be
            aligned (same order and length) with the index vectors.
        query_embedding: A 1-D or 2-D ``numpy.ndarray`` of shape ``(d,)`` or
            ``(1, d)`` containing the query vector.  Must be ``float32``.
        k: Number of nearest neighbours to retrieve.  Clamped to
            ``len(name_list)`` if larger.

    Returns:
        A list of ``(name, distance)`` tuples sorted by ascending L1 distance
        (closest first).

    Raises:
        ValueError: If *query_embedding* has an incompatible dimension.

    Example::

        query_vec = model.encode(["JOHN DOE"], convert_to_numpy=True)[0]
        results = search_index(index, names, query_vec, k=3)
        for name, dist in results:
            print(name, dist)
    """
    k = min(k, len(name_list))
    query = np.array(query_embedding, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    distances, indices = index.search(query, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append((name_list[idx], float(dist)))
    return results
