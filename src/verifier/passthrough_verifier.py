from typing import Any, Dict, List


class PassthroughVerifier:
    def __init__(self, top_k: int = 10, snippet_chars: int = 1000):
        self.top_k = int(top_k)
        self.snippet_chars = int(snippet_chars)

    async def verify(
        self,
        sub_query: str,
        docs: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        selected_docs = docs[: self.top_k]
        evidences_chain = []
        keep_ids = []

        for doc in selected_docs:
            doc_id = str(doc.get("id", ""))
            text = str(doc.get("text", ""))
            if not doc_id:
                continue
            keep_ids.append(doc_id)
            if text:
                evidences_chain.append(f"[{doc_id}]: {text[: self.snippet_chars]}")

        return {
            "sub_query": sub_query,
            "keep_ids": keep_ids,
            "evidences_chain": evidences_chain,
            "sub_query_covered": bool(evidences_chain),
            "verification_disabled": True,
        }
