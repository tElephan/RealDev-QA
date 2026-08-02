from typing import Any, Dict


class IdentityPlanner:
    async def plan(self, query: str) -> Dict[str, Any]:
        return {
            "role": "identity_planner",
            "intent": "not_decomposed",
            "rewritten_query": query,
            "sub_queries": [query],
        }
