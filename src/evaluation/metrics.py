import numpy as np
from typing import List, Set

class Evaluator:
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        intersection = retrieved_k.intersection(ground_truth_ids)
        return len(intersection) / len(ground_truth_ids)

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        if not ground_truth_ids:
            return 0.0
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in ground_truth_ids:
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG
        idcg = 0.0
        for i in range(min(len(ground_truth_ids), k)):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
