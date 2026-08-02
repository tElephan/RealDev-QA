import os
import glob
from typing import List, Dict, Any
from ..utils.helpers import read_jsonl, write_jsonl, append_jsonl, load_json, save_json

class DataLoader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def load_corpus(self) -> List[Dict[str, Any]]:
        # Try standard format first
        path = os.path.join(self.dataset_dir, "corpus.jsonl")
        if os.path.exists(path):
            corpus = read_jsonl(path)
            return corpus

        # Try codeQA format: load from subdirectories
        corpus = []
        subdirs = sorted([d for d in os.listdir(self.dataset_dir)
                         if os.path.isdir(os.path.join(self.dataset_dir, d))
                         and d.isdigit()])

        # Process the first 10 subdirectories
        for subdir in subdirs[:50]:
            subdir_path = os.path.join(self.dataset_dir, subdir)
            # Find all doc_*.txt files
            doc_files = sorted(glob.glob(os.path.join(subdir_path, "doc_*.txt")))

            for doc_file in doc_files:
                doc_id = os.path.basename(doc_file).replace('.txt', '')
                # Read document content
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                corpus.append({
                    "id": f"{subdir}_{doc_id}",
                    "text": content,
                    "query_id": subdir
                })

        return corpus

    def load_queries(self) -> List[Dict[str, Any]]:
        # Try standard format first
        path = os.path.join(self.dataset_dir, "queries.jsonl")
        if os.path.exists(path):
            return read_jsonl(path)

        # Try codeQA format: load from subdirectories
        queries = []
        subdirs = sorted([d for d in os.listdir(self.dataset_dir)
                         if os.path.isdir(os.path.join(self.dataset_dir, d))
                         and d.isdigit()])

        # Process the first 10 subdirectories
        for subdir in subdirs[:50]:
            data_file = os.path.join(self.dataset_dir, subdir, "data.json")
            if os.path.exists(data_file):
                data = load_json(data_file)
                queries.append({
                    "id": subdir,
                    "text": data.get("question_body", ""),
                    "positive_doc_ids": data.get("positive_doc_ids", []),
                    "negative_doc_ids": data.get("negative_doc_ids", [])
                })

        return queries

    def load_relevance(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.dataset_dir, "relevance.jsonl")
        return read_jsonl(path)

    def load_results(self, output_path: str) -> List[Dict[str, Any]]:
        return read_jsonl(output_path)

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        write_jsonl(output_path, results)

    def append_result(self, result: Dict[str, Any], output_path: str):
        append_jsonl(output_path, result)
