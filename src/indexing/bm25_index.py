import os
import json
import subprocess
from typing import List, Dict, Any, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class BM25Index:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.searcher = None

    def build(self, corpus: List[Dict[str, Any]], force: bool = False):
        if os.path.exists(self.index_path) and not force:
            logger.info(f"BM25 index already exists at {self.index_path}. Skipping build.")
            self.load()
            return

        logger.info(f"Building BM25 index at {self.index_path}...")
        temp_dir = os.path.join(os.path.dirname(self.index_path), "temp_bm25")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Prepare files for pyserini
        from tqdm import tqdm
        for i, doc in enumerate(tqdm(corpus, desc="Preparing BM25 docs")):
            with open(os.path.join(temp_dir, f"doc_{i}.json"), "w") as f:
                json.dump({"id": doc["id"], "contents": doc["text"]}, f)

        # Run pyserini indexing command
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", temp_dir,
            "--index", self.index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "16",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        subprocess.run(cmd, check=True)
        logger.info("BM25 index built successfully.")
        self.load()

    def load(self):
        try:
            from pyserini.search.lucene import LuceneSearcher
            self.searcher = LuceneSearcher(self.index_path)
        except ImportError:
            logger.error("pyserini not installed. BM25 search will not work.")

    def search(self, query: str, k: int = 100) -> List[Dict[str, Any]]:
        if not self.searcher:
            return []
        hits = self.searcher.search(query, k=k)
        return [{"id": hit.docid, "score": hit.score} for hit in hits]
