import asyncio
import random
from typing import List, Dict
from openai import AsyncOpenAI
from .base import BaseLLMClient
from ..utils.logger import setup_logger
from ..core.resources import ResourceManager

logger = setup_logger(__name__)

class RemoteLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url

    async def chat(self, messages: List[Dict[str, str]], max_retries: int = 3, **kwargs) -> str:
        sem = ResourceManager.get_instance().llm_semaphore

        last_err = None
        for attempt in range(max_retries):
            try:
                async with sem:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **kwargs,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                last_err = e
                wait_time = (2 ** attempt) + random.random()
                logger.warning(
                    "API call failed (attempt %s/%s) for model=%s base_url=%s: %s. "
                    "Retrying in %.2fs...",
                    attempt + 1,
                    max_retries,
                    self.model,
                    self.base_url,
                    e,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
        
        logger.error(
            "API call failed after %s attempts for model=%s base_url=%s: %s",
            max_retries,
            self.model,
            self.base_url,
            last_err,
        )
        raise last_err
