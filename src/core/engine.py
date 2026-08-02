import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio

from ..models.remote_model import RemoteLLMClient
from ..models.local_model import QwenEmbeddingModel
from ..planner.query_planner import QueryPlanner
from ..planner.identity_planner import IdentityPlanner
from ..indexing.bm25_index import BM25Index
from ..indexing.vector_index import VectorIndex
from ..indexing.entity_index import EntityIndex
from ..indexing.ontology_index import OntologyIndex
from ..search.retriever import MultiModalRetriever
from ..search.ranker import HybridRanker
from ..verifier.verifier import Verifier
from ..verifier.passthrough_verifier import PassthroughVerifier
from ..verifier.reflector import Reflector
from ..utils.logger import setup_logger
from ..io.data_loader import DataLoader
from .resources import ResourceManager

logger = setup_logger(__name__)

class DevRAG:
    """
    DevRAG: an agentic RAG engine with multi-modal retrieval and concept-tree search.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Resource Manager
        ResourceManager.setup(config.get('resources', {}))
        
        self.corpus = {} # Map of doc_id -> text
        self.storage_dir = config['storage']['path']
        self.trace_dir = config.get('storage', {}).get('trace_path', self.storage_dir)
        
        # Initialize Models
        # concurrency settings
        emb_conc = config['search'].get('embedding_concurrency', 1) if 'search' in config else 1
        self.subquery_concurrency = config['search'].get('subquery_concurrency', 4) if 'search' in config else 4

        self.llm_client = RemoteLLMClient(
            api_key=config['model']['api_key'],
            base_url=config['model']['base_url'],
            model=config['model']['llm_name']
        )
        self.embedding_model = QwenEmbeddingModel(
            model_name=config['model']['embedding_name'],
            max_concurrency=emb_conc,
            target_dim=config['model'].get('embedding_dim')
        )
        
        # Initialize Indices
        storage_dir = config['storage']['path']
        os.makedirs(storage_dir, exist_ok=True)
        self.bm25_index = BM25Index(os.path.join(storage_dir, "bm25"))
        self.vector_index = VectorIndex(
            os.path.join(storage_dir, "vector"), 
            dimension=config['model']['embedding_dim'],
            hnsw_m=config['search'].get('faiss', {}).get('hnsw_m', 32),
            ef_construction=config['search'].get('faiss', {}).get('ef_construction', 200)
        )
        self.entity_index = EntityIndex(
            os.path.join(storage_dir, "entity"),
            spacy_model=config['search'].get('spacy', {}).get('model', 'en_core_web_lg')
        )
        self.ontology_index = OntologyIndex(
            os.path.join(storage_dir, "ontology.json"),
            config=config['search'].get('ontology', {})
        )
        # Whether to enable ontology-based search fusion (default: False)
        self.ontology_enabled = bool(config['search'].get('ontology_enabled', False)) if 'search' in config else False
        
        # Initialize Components
        agentic_modules = config.get('agentic_modules', {}) or {}
        ablation_config = config.get('ablation', {}) or {}
        planner_enabled = bool(agentic_modules.get('planner', True))

        if planner_enabled:
            self.planner = QueryPlanner(self.llm_client)
        else:
            logger.info("Planner disabled; using identity planner.")
            self.planner = IdentityPlanner()

        self.retriever = MultiModalRetriever(
            self.bm25_index, self.vector_index, self.entity_index, self.ontology_index,
            self.embedding_model, self.llm_client
        )
        self.ranker = HybridRanker(
            weights=config['search'].get('weights'), 
            concept_bonus=config['search'].get('concept_bonus', 1.5),
            rrf_k=config['search'].get('rrf_k', 60)
        )
        if bool(agentic_modules.get('verifier', True)):
            self.verifier = Verifier(self.llm_client)
        else:
            logger.info("Verifier disabled; using passthrough verifier.")
            self.verifier = PassthroughVerifier(
                top_k=ablation_config.get('passthrough_verifier_top_k', 10),
                snippet_chars=ablation_config.get('passthrough_verifier_snippet_chars', 1000),
            )
        self.reflector = Reflector(self.llm_client)
        if planner_enabled:
            self.reflector_prompt_name = "REFLECTOR_SYSTEM"
            self.reflector_final_prompt_name = "REFLECTOR_FINAL_SYSTEM"
            self.reflector_payload_fields = Reflector.DEFAULT_PAYLOAD_FIELDS
        else:
            self.reflector_prompt_name = "REFLECTOR_NO_PLANNER_SYSTEM"
            self.reflector_final_prompt_name = "REFLECTOR_NO_PLANNER_FINAL_SYSTEM"
            self.reflector_payload_fields = (
                "original_query",
                "accumulated_evidences",
                "sub_queries_status",
            )
        
        self.max_turns = config['search'].get('max_turns', 3)
        self.skip_forced_answer_when_answered = bool(
            config['search'].get('skip_forced_answer_when_answered', False)
        )

    async def build_indices(self, corpus: List[Dict[str, Any]]):
        """Build all indices if they don't exist."""
        self.corpus = {str(doc['id']): doc['text'] for doc in corpus}
        # Build indices in threadpool where operations are blocking
        await asyncio.to_thread(self.bm25_index.build, corpus)
        await asyncio.to_thread(self.vector_index.build, corpus, self.embedding_model)
        await asyncio.to_thread(self.entity_index.build, corpus)

        # Try to reuse precomputed embeddings from vector index to avoid recomputing
        doc_vecs = self.vector_index.load_embeddings()
        if doc_vecs is None:
            # fallback: compute via embedding model
            encode_fn = getattr(self.embedding_model, "encode_async", None)
            texts = [doc["text"] for doc in corpus]
            if encode_fn is None:
                doc_vecs = await asyncio.to_thread(self.embedding_model.encode, texts)
            else:
                doc_vecs = await encode_fn(texts)

        if not self.ontology_enabled:
            logger.info("Ontology search is disabled; skipping ontology index build.")
            return

        force_rebuild_ontology = bool(self.config.get('storage', {}).get('force_rebuild_ontology', False))
        await self.ontology_index.build(
            corpus,
            self.llm_client,
            self.embedding_model,
            doc_vecs=doc_vecs,
            force=force_rebuild_ontology,
        )

    async def search(self, query: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform agentic multi-turn search."""
        current_query = query
        turn = 0
        all_evidences = []
        all_doc_ids = set()
        trace_queries: List[str] = []
        turn_trace_details: List[Dict[str, Any]] = []
        forced_answers_by_turn: List[Dict[str, Any]] = []
        last_reflection = None
        last_forced_answer = None

        while turn < self.max_turns:
            logger.info(f"Turn {turn + 1}: Planning for query: {current_query}")
            plan = await self.planner.plan(current_query)
            sub_queries = plan.get("sub_queries", [current_query])
            # record the query used in this turn
            trace_queries.append(current_query)
            # Process sub-queries with controlled concurrency
            sem = asyncio.Semaphore(self.subquery_concurrency)

            async def _process_subq(sub_q: str):
                # limit number of parallel retrieve calls
                async with sem:
                    multi_results = await self.retriever.retrieve(sub_q, use_ontology=self.ontology_enabled)

                ranked_docs = self.ranker.rank(multi_results)

                # Fetch full text for verification
                docs_for_verify = []
                for doc_res in ranked_docs:
                    doc_id = str(doc_res['id'])
                    if doc_id in self.corpus:
                        docs_for_verify.append({"id": doc_id, "text": self.corpus[doc_id]})

                verification = await self.verifier.verify(sub_q, docs_for_verify)
                verification["sub_query"] = sub_q
                return verification

            tasks = [_process_subq(sq) for sq in sub_queries]
            turn_results = await asyncio.gather(*tasks)
            turn_detail = {
                "turn": turn + 1,
                "main_query": current_query,
                "sub_queries": [
                    {
                        "sub_query": verification.get("sub_query", ""),
                        "verifier_result": verification,
                    }
                    for verification in turn_results
                ],
            }
            turn_trace_details.append(turn_detail)

            for verification in turn_results:
                all_evidences.extend(verification.get("evidences_chain", []))
                all_doc_ids.update(verification.get("keep_ids", []))
            
            # Reflect on the current turn's results using the query that was planned/executed this turn
            # Provide original query, current (rewritten) query, verifier results and accumulated evidences
            reflection = await self.reflector.reflect(
                query,
                current_query,
                turn_results,
                all_evidences,
                prompt_name=self.reflector_prompt_name,
                payload_fields=self.reflector_payload_fields,
            )
            last_reflection = reflection
            turn_detail["reflection"] = reflection

            if reflection.get("answered") and self.skip_forced_answer_when_answered:
                logger.info("Query fully answered; skipping forced answer because skip_forced_answer_when_answered=true.")
                turn_detail["forced_answer_skipped"] = True
                try:
                    self._persist_turn_answer(
                        request_id,
                        original_query=query,
                        turn=turn + 1,
                        main_query=current_query,
                        answer_payload=reflection,
                        source="reflection",
                        forced_answer_skipped=True,
                    )
                except Exception:
                    logger.warning("Failed to write turn answer for answered request")
                # persist trace for answered requests as well
                try:
                    self._persist_query_traces(request_id, trace_queries, turn_trace_details)
                except Exception:
                    logger.warning("Failed to write query trace for answered request")

                result = {
                    "query": query,
                    "answer": reflection.get("final_answer"),
                    "evidences": all_evidences,
                    "thought": reflection.get("thought"),
                    "doc_ids": list(all_doc_ids),
                    "turns": turn + 1,
                    "forced_answers": forced_answers_by_turn,
                }
                try:
                    self._persist_answer(request_id, result)
                except Exception:
                    logger.warning("Failed to write answer for answered request")
                return result

            forced_answer = await self.reflector.force_answer(
                query,
                current_query,
                turn_results,
                all_evidences,
                prompt_name=self.reflector_final_prompt_name,
                payload_fields=self.reflector_payload_fields,
            )
            last_forced_answer = forced_answer
            forced_answer_record = {
                "turn": turn + 1,
                "main_query": current_query,
                "forced_answer": forced_answer,
            }
            forced_answers_by_turn.append(forced_answer_record)
            turn_detail["forced_answer"] = forced_answer
            try:
                self._persist_turn_answer(
                    request_id,
                    original_query=query,
                    turn=turn + 1,
                    main_query=current_query,
                    answer_payload=forced_answer,
                    source="forced_answer",
                    forced_answer_skipped=False,
                )
            except Exception:
                logger.warning("Failed to write turn forced answer")

            forced_answer_text = str(forced_answer.get("final_answer", "")).replace("\n", " ")
            logger.info("Turn %s forced answer: %s", turn + 1, forced_answer_text[:500])

            if reflection.get("answered"):
                logger.info("Query fully answered.")
                # persist trace for answered requests as well
                try:
                    self._persist_query_traces(request_id, trace_queries, turn_trace_details)
                except Exception:
                    logger.warning("Failed to write query trace for answered request")

                result = {
                    "query": query,
                    "answer": reflection.get("final_answer"),
                    "evidences": all_evidences,
                    "thought": reflection.get("thought"),
                    "doc_ids": list(all_doc_ids),
                    "turns": turn + 1,
                    "forced_answers": forced_answers_by_turn,
                }
                try:
                    self._persist_answer(request_id, result)
                except Exception:
                    logger.warning("Failed to write answer for answered request")
                return result
            # Update current_query to the new query suggested by reflector (defaults to previous current_query)
            current_query = reflection.get("new_query", current_query)
            turn += 1

        forced_final = last_forced_answer
        if forced_final is None:
            forced_final = await self.reflector.force_answer(
                query,
                current_query,
                turn_results if 'turn_results' in locals() else [],
                all_evidences,
                prompt_name=self.reflector_final_prompt_name,
                payload_fields=self.reflector_payload_fields,
            )
            forced_answers_by_turn.append({
                "turn": turn,
                "main_query": current_query,
                "forced_answer": forced_final,
            })
            try:
                self._persist_turn_answer(
                    request_id,
                    original_query=query,
                    turn=turn,
                    main_query=current_query,
                    answer_payload=forced_final,
                    source="forced_final",
                    forced_answer_skipped=False,
                )
            except Exception:
                logger.warning("Failed to write final forced turn answer")
        final_ans = (
            forced_final.get("final_answer")
            or (last_reflection.get("final_answer") if last_reflection else None)
            or "Unknown"
        )
        final_thought = (
            forced_final.get("thought")
            or (last_reflection.get("thought") if last_reflection else None)
        )
        # persist trace: one line per request with id first
        try:
            self._persist_query_traces(request_id, trace_queries, turn_trace_details)
        except Exception:
            logger.warning("Failed to write query trace")
        result = {
            "query": query,
            "answer": final_ans,
            "evidences": all_evidences,
            "thought": final_thought,
            "doc_ids": list(all_doc_ids),
            "turns": turn,
            "forced_answers": forced_answers_by_turn,
        }
        try:
            self._persist_answer(request_id, result)
        except Exception:
            logger.warning("Failed to write answer")
        return result

    def _persist_answer(
        self,
        request_id: Optional[str],
        result: Dict[str, Any],
    ) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        rid = str(request_id) if request_id is not None else "unknown"

        answer_path = os.path.join(self.trace_dir, "answers.jsonl")
        answer_record = {
            "id": rid,
            "record_type": "final",
            "original_query": result.get("query"),
            "query": result.get("query"),
            "answer": result.get("answer"),
            "thought": result.get("thought"),
            "turns": result.get("turns"),
            "doc_ids": result.get("doc_ids", []),
        }
        with open(answer_path, "a", encoding="utf-8") as af:
            af.write(json.dumps(answer_record, ensure_ascii=False) + "\n")

    def _persist_turn_answer(
        self,
        request_id: Optional[str],
        original_query: str,
        turn: int,
        main_query: str,
        answer_payload: Dict[str, Any],
        source: str,
        forced_answer_skipped: bool,
    ) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        rid = str(request_id) if request_id is not None else "unknown"

        answer_path = os.path.join(self.trace_dir, "answers.jsonl")
        answer_record = {
            "id": rid,
            "record_type": "turn",
            "original_query": original_query,
            "turn": turn,
            "main_query": main_query,
            "answer": answer_payload.get("final_answer"),
            "thought": answer_payload.get("thought"),
            "source": source,
            "forced_answer_skipped": forced_answer_skipped,
        }
        with open(answer_path, "a", encoding="utf-8") as af:
            af.write(json.dumps(answer_record, ensure_ascii=False) + "\n")

    def _persist_query_traces(
        self,
        request_id: Optional[str],
        trace_queries: List[str],
        turn_trace_details: List[Dict[str, Any]],
    ) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        rid = str(request_id) if request_id is not None else "unknown"

        trace_path = os.path.join(self.trace_dir, "query_traces.jsonl")
        with open(trace_path, "a", encoding="utf-8") as tf:
            ordered = {"id": rid, "queries": trace_queries}
            tf.write(json.dumps(ordered, ensure_ascii=False) + "\n")

        detail_trace_path = os.path.join(self.trace_dir, "query_turn_details.jsonl")
        with open(detail_trace_path, "a", encoding="utf-8") as tf:
            ordered = {"id": rid, "turns": turn_trace_details}
            tf.write(json.dumps(ordered, ensure_ascii=False) + "\n")
