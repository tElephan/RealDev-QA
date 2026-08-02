from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> Any:
        pass

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass
