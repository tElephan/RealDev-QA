import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from ..indexing.bm25_index import BM25Index
from ..indexing.vector_index import VectorIndex
from ..indexing.entity_index import EntityIndex
from ..indexing.ontology_index import OntologyIndex
from ..models.base import BaseEmbeddingModel, BaseLLMClient
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MultiModalRetriever:
    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        entity_index: EntityIndex,
        ontology_index: OntologyIndex,
        embedding_model: BaseEmbeddingModel,
        llm_client: BaseLLMClient
    ):
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.entity_index = entity_index
        self.ontology_index = ontology_index
        self.embedding_model = embedding_model
        self.llm_client = llm_client

    async def retrieve(self, query: str, top_k: int = 100, use_ontology: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        # 1. BM25
        # Use async-safe wrappers for potentially blocking operations
        encode_fn = getattr(self.embedding_model, "encode_async", None)

        if encode_fn is None:
            # fallback to thread executor
            async def _encode(texts, is_query=True):
                return await asyncio.to_thread(self.embedding_model.encode, texts, is_query)
        else:
            async def _encode(texts, is_query=True):
                return await encode_fn(texts, is_query=is_query)

        # 1. compute query embedding (may be blocking -> run in thread or encode_async)
        query_emb = await _encode([query], is_query=True)
        # some implementations return array, take first
        try:
            qemb = query_emb[0]
        except Exception:
            qemb = query_emb

        # 2. Launch synchronous index searches in parallel using threadpool where appropriate
        tasks = []
        tasks.append(asyncio.to_thread(self.bm25_index.search, query, top_k))
        tasks.append(asyncio.to_thread(self.vector_index.search, qemb, top_k))
        tasks.append(asyncio.to_thread(self.entity_index.search, query, top_k))
        # Ontology search is async (and will use llm/embedder internally)
        if use_ontology:
            tasks.append(self.ontology_index.search(
                query,
                self.embedding_model,
                self.llm_client,
                top_k_nodes=self.ontology_index.config.get('search_top_k_nodes', 20)
            ))

        gathered = await asyncio.gather(*tasks)

        # unpack results
        bm25_results, vector_results, entity_results = gathered[0], gathered[1], gathered[2]
        ontology_results = []
        ontology_trace = None
        if use_ontology and len(gathered) > 3:
            ontology_payload = gathered[3]
            if isinstance(ontology_payload, dict):
                ontology_results = ontology_payload.get("results", []) or []
                ontology_trace = ontology_payload.get("trace")
            else:
                ontology_results = ontology_payload or []

        return {
            "bm25": bm25_results,
            "vector": vector_results,
            "entity": entity_results,
            "ontology": ontology_results,
            "ontology_trace": ontology_trace,
        }
