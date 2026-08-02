import os
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from ..utils.logger import setup_logger
from ..models.base import BaseEmbeddingModel

logger = setup_logger(__name__)

class VectorIndex:
    def __init__(self, index_path: str, dimension: int = 1536, hnsw_m: int = 32, ef_construction: int = 200):
        self.index_path = index_path
        self.dimension = dimension
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.index = None
        self.doc_ids = []

    def build(self, corpus: List[Dict[str, Any]], model: BaseEmbeddingModel, batch_size: int = 4, force: bool = False):
        ids_path = self.index_path + ".ids.json"
        if os.path.exists(self.index_path) and os.path.exists(ids_path) and not force:
            logger.info(f"Vector index already exists at {self.index_path}. Skipping build.")
            self.load()
            return

        logger.info(f"Building Vector index at {self.index_path}...")
        texts = [doc["text"] for doc in corpus]
        self.doc_ids = [doc["id"] for doc in corpus]
        
        # Initialize FAISS HNSW index
        self.index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Encode in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus embeddings"):
            batch_texts = texts[i:i+batch_size]
            embeddings = model.encode(batch_texts)
            all_embeddings.append(embeddings)

        full_embeddings = np.vstack(all_embeddings).astype('float32')
        full_embeddings = np.ascontiguousarray(full_embeddings)
        # Validate dimensions against configured index dimension
        if full_embeddings.shape[1] < self.dimension:
            raise ValueError(f"Embeddings dim {full_embeddings.shape[1]} < index dimension {self.dimension}")
        if full_embeddings.shape[1] > self.dimension:
            # Truncate to configured dimension
            full_embeddings = full_embeddings[:, : self.dimension]
        faiss.normalize_L2(full_embeddings)
        self.index.add(full_embeddings)

        # persist embeddings for reuse by other components (e.g., ontology)
        embs_path = self.index_path + ".embs.npy"
        try:
            np.save(embs_path, full_embeddings)
        except Exception:
            logger.warning(f"Failed to save embeddings to {embs_path}")

        faiss.write_index(self.index, self.index_path)
        import json
        with open(ids_path, "w") as f:
            json.dump(self.doc_ids, f)
        logger.info("Vector index built successfully.")

    def load_embeddings(self) -> Optional[np.ndarray]:
        """Load persisted embeddings if available."""
        embs_path = self.index_path + ".embs.npy"
        if os.path.exists(embs_path):
            try:
                return np.load(embs_path)
            except Exception:
                logger.warning(f"Failed to load embeddings from {embs_path}")
        return None

    def load(self):
        self.index = faiss.read_index(self.index_path)
        ids_path = self.index_path + ".ids.json"
        import json
        with open(ids_path, "r") as f:
            self.doc_ids = json.load(f)

    def search(self, query_embedding: np.ndarray, k: int = 100, ef_search: int = 64) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        
        # Set efSearch if it's an HNSW index
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = ef_search
            
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        query_embedding = np.ascontiguousarray(query_embedding)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({"id": self.doc_ids[idx], "score": float(score)})
        return results
