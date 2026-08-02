from typing import List, Dict, Any
from collections import defaultdict

class HybridRanker:
    def __init__(self, weights: Dict[str, float] = None, concept_bonus: float = 1.5, rrf_k: int = 60):
        self.weights = weights or {
            "bm25": 0.3,
            "vector": 1.0,
            "entity": 0.2,
            "ontology": 1.0
        }
        self.concept_bonus = concept_bonus
        self.rrf_k = rrf_k

    def _rrf(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate RRF scores for a single retrieval result list."""
        # Sort by score descending to get ranks
        ranked = sorted(results, key=lambda x: x["score"], reverse=True)
        return {
            res["id"]: 1.0 / (self.rrf_k + rank + 1)
            for rank, res in enumerate(ranked)
        }

    def rank(self, multi_results: Dict[str, List[Dict[str, Any]]], top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. Calculate RRF for base methods
        base_methods = ["bm25", "vector", "entity"]
        rrf_scores = {}
        for method in base_methods:
            if method in multi_results and multi_results[method]:
                rrf_scores[method] = self._rrf(multi_results[method])
        
        # 2. Fuse base scores
        all_doc_ids = set().union(*[scores.keys() for scores in rrf_scores.values()])
        fused_scores = defaultdict(float)
        for doc_id in all_doc_ids:
            for method, scores in rrf_scores.items():
                fused_scores[doc_id] += scores.get(doc_id, 0.0) * self.weights.get(method, 1.0)
        
        # 3. Add Ontology Bonus
        if "ontology" in multi_results and multi_results["ontology"]:
            ontology_results = multi_results["ontology"]
            for res in ontology_results:
                doc_id = res["id"]
                # Apply weight and bonus directly to the raw score (1 or 2)
                bonus = res["score"] * self.weights.get("ontology", 1.0) * self.concept_bonus
                fused_scores[doc_id] += bonus

        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"id": doc_id, "score": score} for doc_id, score in ranked_ids[:top_k]]
