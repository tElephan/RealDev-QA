import json
from typing import List, Dict, Any
from ..models.base import BaseLLMClient
from ..prompts.manager import PromptManager
from ..utils.helpers import parse_llm_json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Verifier:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    async def verify(self, sub_query: str, docs: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
        system_prompt = PromptManager.get_prompt('VERIFIER_SYSTEM')
        
        all_evidences_chain = []
        keep_ids = set()
        covered = False

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            payload = {
                "sub_query": sub_query,
                "context_docs": [{"id": d["id"], "text": d["text"][:8000]} for d in batch]
            }
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ]
            
            try:
                response_text = await self.llm_client.chat(messages, response_format={"type": "json_object"})
                res = parse_llm_json(response_text)
                if not res:
                    continue
                
                # Extract keep_ids
                for did in res.get("keep_ids", []) or []:
                    keep_ids.add(str(did))

                # Extract and format evidences
                entries = res.get("evidences_chain", []) or []
                for ev in entries:
                    if isinstance(ev, dict):
                        fact = ev.get("fact", "").strip()
                        sid = ev.get("source_id", "")
                        if fact:
                            formatted_ev = f"[{sid}]: {fact}" if sid else fact
                            all_evidences_chain.append(formatted_ev)
                            if sid:
                                keep_ids.add(str(sid))

                if res.get("sub_query_covered") is True:
                    covered = True
                    break
                    
            except Exception as e:
                logger.error(f"Verification batch failed: {e}")
                continue
        
        return {
            "sub_query": sub_query,
            "keep_ids": list(keep_ids),
            "evidences_chain": all_evidences_chain,
            "sub_query_covered": covered
        }
