import os
import json
import spacy
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class SpacyExtractor:
    def __init__(self, model_name: str = "en_core_web_lg"):
        try:
            self.nlp = spacy.load(model_name)
        except Exception:
            logger.info(f"Downloading spacy model {model_name}...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        # Increase max_length to handle large documents
        self.nlp.max_length = 10000000

        if "parser" not in self.nlp.pipe_names and "senter" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def _extract_from_doc(self, doc: "spacy.tokens.Doc") -> Dict[str, Any]:
        all_terms = set()
        double_nouns = {}
        appearance_count = {}

        for sent in doc.sents:
            ent_positions = set()
            sentence_terms = []

            for ent in sent.ents:
                ent_text = ent.text.strip().lower()
                if not ent_text:
                    continue
                sentence_terms.append(ent_text)
                if ent.label_ == "PERSON":
                    parts = [p.strip().lower() for p in ent_text.split() if p.strip()]
                    if len(parts) >= 2:
                        double_nouns[ent_text] = parts
                for tok in ent:
                    ent_positions.add(tok.i)

            tokens = list(sent)
            for i, token in enumerate(tokens):
                if token.i in ent_positions:
                    continue
                if token.pos_ == "ADJ" and i + 1 < len(tokens):
                    nxt = tokens[i + 1]
                    if nxt.pos_ == "NOUN":
                        phrase = f"{token.lemma_.lower()} {nxt.lemma_.lower()}".strip()
                        if phrase:
                            sentence_terms.append(phrase)
                        continue
                if token.pos_ in ("NOUN", "PROPN") and token.lemma_.strip():
                    term = token.lemma_.lower()
                    sentence_terms.append(term)

            for t in sentence_terms:
                all_terms.add(t)
                appearance_count[t] = appearance_count.get(t, 0) + 1

        return {
            "nouns": sorted(all_terms),
            "double_nouns": double_nouns,
            "appearance_count": appearance_count,
        }

    def extract(self, text: str) -> Dict[str, Any]:
        return self._extract_from_doc(self.nlp(text))

    def extract_batch(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            results.append(self._extract_from_doc(doc))
        return results

class EntityIndex:
    def __init__(self, index_path: str, spacy_model: str = "en_core_web_lg"):
        self.index_path = index_path
        self.extractor = SpacyExtractor(spacy_model)
        self.entity_map = defaultdict(list)
        self.appearance_count = defaultdict(int)

    def build(self, corpus: List[Dict[str, Any]], force: bool = False):
        appc_path = self.index_path + ".appc.json"
        if os.path.exists(self.index_path) and os.path.exists(appc_path) and not force:
            logger.info(f"Entity index already exists at {self.index_path}. Skipping build.")
            self.load()
            return

        logger.info(f"Building Entity index at {self.index_path}...")
        for doc in tqdm(corpus, desc="Extracting entities"):
            res = self.extractor.extract(doc["text"])
            nouns = res.get("nouns", [])
            double_n = res.get("double_nouns", {})
            appc = res.get("appearance_count", {})

            for noun in nouns:
                if doc["id"] not in self.entity_map[noun]:
                    self.entity_map[noun].append(doc["id"])
            
            for fullname, parts in double_n.items():
                for p in parts:
                    if doc["id"] not in self.entity_map[p]:
                        self.entity_map[p].append(doc["id"])
                    self.appearance_count[p] += 1
            
            for k, v in appc.items():
                self.appearance_count[k] += int(v)

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.entity_map), f, ensure_ascii=False)
        with open(appc_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.appearance_count), f, ensure_ascii=False)
        logger.info("Entity index built successfully.")

    def load(self):
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.entity_map = json.load(f)
        appc_path = self.index_path + ".appc.json"
        with open(appc_path, "r", encoding="utf-8") as f:
            self.appearance_count = json.load(f)

    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        q_res = self.extractor.extract(query)
        q_nouns = q_res.get("nouns", [])
        q_double = q_res.get("double_nouns", {})

        counter = defaultdict(int)
        for term in q_nouns:
            for did in self.entity_map.get(term, []):
                counter[did] += 1
        
        for fullname, parts in q_double.items():
            for did in self.entity_map.get(fullname, []):
                counter[did] += 1
            for p in parts:
                for did in self.entity_map.get(p, []):
                    counter[did] += 1
        
        results = [{"id": doc_id, "score": count} for doc_id, count in counter.items()]
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
