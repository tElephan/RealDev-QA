import numpy as np
import asyncio
import faiss
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from .ontology_index import TreeNode
from ..models.base import BaseLLMClient, BaseEmbeddingModel
from ..prompts.manager import PromptManager
from ..utils.helpers import parse_llm_json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class KMeans:
    def __init__(self, k: int, niter: int = 5):
        self.k = k
        self.niter = niter

    def _normalize(self, X):
        norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norm

    def run(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, d = x.shape
        x = x.astype('float32')
        x = self._normalize(x)
        
        # Try GPU first
        try:
            res = faiss.StandardGpuResources()
            kmeans = faiss.Kmeans(d, self.k, niter=self.niter, verbose=False, gpu=True)
            kmeans.train(x)
            D, I = kmeans.index.search(x, 1)
            return I.reshape(-1), kmeans.centroids
        except Exception:
            # Fallback to CPU
            kmeans = faiss.Kmeans(d, self.k, niter=self.niter, verbose=False, gpu=False)
            kmeans.train(x)
            D, I = kmeans.index.search(x, 1)
            return I.reshape(-1), kmeans.centroids

class Selector:
    @staticmethod
    def medoids(x: np.ndarray, labels: np.ndarray, top_k: int = 5) -> Dict[int, List[int]]:
        """Find indices of documents closest to cluster centroids."""
        unique_labels = np.unique(labels)
        res = {}
        for label in unique_labels:
            if label == -1: continue
            idx = np.where(labels == label)[0]
            if len(idx) == 0: continue
            
            cluster_points = x[idx]
            centroid = np.mean(cluster_points, axis=0)
            
            # Euclidean distance to centroid
            dists = np.sum((cluster_points - centroid) ** 2, axis=1)
            rel_idx = np.argsort(dists)[:top_k]
            res[int(label)] = idx[rel_idx].tolist()
        return res

class Router:
    @staticmethod
    def route(doc_embs: np.ndarray, cat_embs: np.ndarray, threshold: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
        # Cosine similarity
        sims = doc_embs @ cat_embs.T
        
        assignment = []
        max_scores = []

        for i in range(len(sims)):
            cid = np.argmax(sims[i])
            score = sims[i][cid]
            max_scores.append(score)

            if score < threshold:
                assignment.append(-1)
            else:
                assignment.append(cid)

        return np.array(assignment), np.array(max_scores)

class TreeBuilder:
    def __init__(
        self,
        llm: BaseLLMClient,
        embedder: BaseEmbeddingModel,
        max_depth: int = 4,
        min_cluster_k: int = 10,
        max_cluster_k: int = 60,
        min_docs_to_split: int = 100,
        concurrency: int = 16,
        sample_docs_for_split: int = 10,
        route_threshold: float = 0.45,
    ):
        self.llm = llm
        self.embedder = embedder
        self.max_depth = max_depth
        self.min_cluster_k = min_cluster_k
        self.max_cluster_k = max_cluster_k
        self.min_docs_to_split = min_docs_to_split
        self.sample_docs_for_split = sample_docs_for_split
        self.route_threshold = route_threshold
        # Semaphore created lazily per event loop to avoid binding to wrong loop
        self._sem = None
        self._concurrency = concurrency

    def _ensure_sem(self) -> asyncio.Semaphore:
        """Ensure a semaphore exists for the current running event loop."""
        loop = asyncio.get_running_loop()
        sem = getattr(self, "_sem", None)
        sem_loop = getattr(self, "_sem_loop", None)
        # create/recreate semaphore for the current loop if needed
        if sem is None or sem_loop is not loop:
            sem = asyncio.Semaphore(self._concurrency)
            self._sem = sem
            self._sem_loop = loop
        return sem

    def choose_raw_k(self, N: int) -> int:
        k = int(np.sqrt(N / 2))
        k = max(self.min_cluster_k, k)
        k = min(self.max_cluster_k, k)
        if N <= 50:
            k = 4
        max_reasonable = max(2, N // 6)
        k = min(k, max_reasonable)
        return max(2, k)

    async def build_layer(self, docs: List[str], ids: List[str], X: np.ndarray, depth: int, parent_chain: List[str]) -> TreeNode:
        current_name = parent_chain[-1]
        node = TreeNode(name=current_name, level=depth + 1)
        
        # Leaf condition
        if depth >= self.max_depth or len(docs) < self.min_docs_to_split:
            # logger.info(f"Depth {depth}: Leaf reached for {current_name} ({len(docs)} docs)")
            node.doc_ids = ids
            return node

        N = len(docs)
        raw_k = self.choose_raw_k(N)
        raw_k = min(raw_k, N - 1)
        if raw_k <= 1:
            # logger.info(f"Depth {depth}: Too few clusters for {current_name}")
            node.doc_ids = ids
            return node

        # 1. Clustering
        labels, _ = KMeans(raw_k).run(X)
        reps = Selector.medoids(X, labels, top_k=min(self.sample_docs_for_split, 12))
        cluster_sizes = {
            int(cid): int(np.sum(labels == cid))
            for cid in np.unique(labels)
            if cid != -1
        }

        cluster_samples = {
            cid: [docs[i] for i in idx_list]
            for cid, idx_list in reps.items()
        }

        # 2. Generate raw topics (Sequential to match SelectRAG exactly)
        parent_path = " > ".join(parent_chain)
        raw_topics = []
        for cid in tqdm(sorted(cluster_samples.keys()), desc=f"Generating topics depth={depth}"):
            samples = cluster_samples[cid]
            sample_text = "\n\n".join([s[:500] for s in samples])
            prompt = PromptManager.get_prompt('TREE_GEN_TOPIC').format(
                parent_path=parent_path,
                sample_text=sample_text,
                cluster_doc_count=cluster_sizes.get(cid, len(samples)),
            )
            sem = self._ensure_sem()
            async with sem:
                res = await self.llm.chat([{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            
            try:
                data = parse_llm_json(res) or {}
                name = data.get("name", f"Subtopic {cid}")
                desc = data.get("description", "")
                raw_topics.append({
                    "name": name,
                    "description": desc,
                    "cid": cid,
                    "doc_count": cluster_sizes.get(cid, 0),
                })
            except Exception as e:
                logger.warning(f"Failed to parse topic for cluster {cid}: {e}")
                raw_topics.append({
                    "name": f"Subtopic {cid}",
                    "description": "",
                    "cid": cid,
                    "doc_count": cluster_sizes.get(cid, 0),
                })

        if len(raw_topics) <= 1:
            node.doc_ids = ids
            return node

        cat_names = [topic["name"] for topic in raw_topics]
        cat_descs = [topic["description"] for topic in raw_topics]
        encode_fn = getattr(self.embedder, "encode_async", None)
        cat_embs_list = []
        for i in tqdm(range(len(cat_names)), desc="Embedding topic names"):
            text = f"{parent_path} > {cat_names[i]} — {cat_descs[i]}"
            if encode_fn is None:
                emb = await asyncio.to_thread(self.embedder.encode, [text])
            else:
                emb = await encode_fn([text])
            cat_embs_list.append(emb[0])
        cat_embs = np.vstack(cat_embs_list)
        faiss.normalize_L2(cat_embs)

        assignment, _ = Router.route(X, cat_embs, threshold=self.route_threshold)
        valid_child_ids = [cid for cid in range(len(cat_names)) if np.any(assignment == cid)]
        if len(valid_child_ids) <= 1:
            node.doc_ids = ids
            return node

        for cid in valid_child_ids:
            idx = np.where(assignment == cid)[0]
            if len(idx) == 0:
                continue

            sub_docs = [docs[i] for i in idx]
            sub_ids = [ids[i] for i in idx]
            sub_X = X[idx]

            child_chain = parent_chain + [cat_names[cid]]
            child_node = await self.build_layer(sub_docs, sub_ids, sub_X, depth + 1, child_chain)
            if not child_node.desc:
                child_node.desc = cat_descs[cid]
            node.add_child(child_node)

        # Documents not assigned to any child stay at this node
        unassigned_idx = np.where(assignment == -1)[0]
        for i in unassigned_idx:
            node.add_doc(ids[i])

        return node

    async def build(self, docs: List[str], ids: List[str], X: np.ndarray) -> TreeNode:
        logger.info(f"Starting hierarchical tree building for {len(docs)} documents...")
        return await self.build_layer(docs, ids, X, depth=0, parent_chain=["ROOT"])
