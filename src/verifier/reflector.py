import json
from typing import List, Dict, Any, Optional, Sequence
from ..models.base import BaseLLMClient
from ..prompts.manager import PromptManager
from ..utils.helpers import parse_llm_json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Reflector:
    DEFAULT_PAYLOAD_FIELDS = (
        "original_query",
        "current_query",
        "accumulated_evidences",
        "sub_queries_status",
    )

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def _build_payload(
        self,
        original_query: Optional[str],
        current_query: str,
        verifier_results: List[Dict[str, Any]],
        accumulated_evidences: List[str],
        payload_fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        payload_fields = tuple(payload_fields or self.DEFAULT_PAYLOAD_FIELDS)
        payload: Dict[str, Any] = {}

        if "original_query" in payload_fields and original_query is not None:
            payload["original_query"] = original_query
        if "current_query" in payload_fields:
            payload["current_query"] = current_query
        if "accumulated_evidences" in payload_fields:
            payload["accumulated_evidences"] = accumulated_evidences
        if "sub_queries_status" in payload_fields:
            payload["sub_queries_status"] = [
                {
                    "q": res["sub_query"],
                    "facts": res.get("evidences_chain", [])
                }
                for res in verifier_results
            ]

        return payload

    async def reflect(
        self,
        original_query: str,
        current_query: str,
        verifier_results: List[Dict[str, Any]],
        accumulated_evidences: List[str],
        prompt_name: str = "REFLECTOR_SYSTEM",
        payload_fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Reflector now receives both the original user query and the current (possibly rewritten) query,
        along with verifier results and a global list of accumulated evidences. This allows global reasoning
        and proposing a new_query that better targets remaining gaps.
        """
        system_prompt = PromptManager.get_prompt(prompt_name)
        payload = self._build_payload(
            original_query,
            current_query,
            verifier_results,
            accumulated_evidences,
            payload_fields=payload_fields,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        
        try:
            response_text = await self.llm_client.chat(messages, response_format={"type": "json_object"})
            data = parse_llm_json(response_text)
            if not data:
                return {"answered": False, "new_query": current_query}

            return data
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {"answered": False, "new_query": current_query}

    async def force_answer(
        self,
        original_query: str,
        current_query: str,
        verifier_results: List[Dict[str, Any]],
        accumulated_evidences: List[str],
        prompt_name: str = "REFLECTOR_FINAL_SYSTEM",
        payload_fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        system_prompt = PromptManager.get_prompt(prompt_name)
        payload = self._build_payload(
            original_query,
            current_query,
            verifier_results,
            accumulated_evidences,
            payload_fields=payload_fields,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]

        try:
            response_text = await self.llm_client.chat(messages, response_format={"type": "json_object"})
            data = parse_llm_json(response_text)
            if not data:
                return {
                    "final_answer": "Unable to determine a complete answer from the retrieved evidence.",
                    "thought": "No valid JSON was returned during forced final synthesis.",
                }
            return data
        except Exception as e:
            logger.error(f"Forced final answer synthesis failed: {e}")
            return {
                "final_answer": "Unable to determine a complete answer from the retrieved evidence.",
                "thought": f"Forced final synthesis failed: {e}",
            }
