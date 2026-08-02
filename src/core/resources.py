import asyncio
from typing import Dict, Any, Optional

class ResourceManager:
    _instance = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._llm_sem: Optional[asyncio.Semaphore] = None
        self._gpu_sem: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Default initialization if setup wasn't called
            cls._instance = cls({})
        return cls._instance
    
    @classmethod
    def setup(cls, config: Dict[str, Any]):
        """Initialize the singleton with configuration."""
        cls._instance = cls(config)

    def _check_loop(self):
        """Ensure semaphores are bound to the current running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return 
            
        if self._loop is not loop:
            self._loop = loop
            # Re-create semaphores for the new loop
            llm_limit = self.config.get('global_llm_concurrency', 16)
            gpu_limit = self.config.get('global_gpu_concurrency', 1)
            self._llm_sem = asyncio.Semaphore(llm_limit)
            self._gpu_sem = asyncio.Semaphore(gpu_limit)

    @property
    def llm_semaphore(self) -> asyncio.Semaphore:
        self._check_loop()
        if self._llm_sem is None:
             # Fallback if _check_loop didn't run (e.g. no loop yet), though unlikely in usage
             self._llm_sem = asyncio.Semaphore(self.config.get('global_llm_concurrency', 16))
        return self._llm_sem

    @property
    def gpu_semaphore(self) -> asyncio.Semaphore:
        self._check_loop()
        if self._gpu_sem is None:
             self._gpu_sem = asyncio.Semaphore(self.config.get('global_gpu_concurrency', 1))
        return self._gpu_sem
