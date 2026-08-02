import json
from typing import List, Dict, Any, Optional
from ..models.base import BaseLLMClient
from ..prompts.manager import PromptManager
from ..utils.helpers import parse_llm_json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryPlanner:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    async def plan(self, query: str) -> Dict[str, Any]:
        system_prompt = PromptManager.get_prompt('QUERY_PLANNER_SYSTEM')
        user_prompt = f"User query:\n{query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response_text = await self.llm_client.chat(messages, response_format={"type": "json_object"})
            data = parse_llm_json(response_text)

            if not data:
                logger.warning("Failed to parse JSON from planner response. Using raw query.")
                return {
                    "role": "unknown",
                    "intent": "unknown",
                    "rewritten_query": query,
                    "sub_queries": [query]
                }
            
            # Ensure sub_queries is a list
            if "sub_queries" not in data or not data["sub_queries"]:
                data["sub_queries"] = [data.get("rewritten_query", query)]
            
            return data
        except Exception as e:
            logger.error(f"Query planning failed: {e}")
            return {
                "role": "error",
                "intent": "error",
                "rewritten_query": query,
                "sub_queries": [query]
            }
