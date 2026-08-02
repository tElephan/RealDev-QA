import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import asyncio
import os
from src.core.engine import DevRAG
from src.io.data_loader import DataLoader
from src.utils.logger import setup_logger
from tqdm import tqdm

# Temporarily silence noisy httpx/httpcore logs (option 1)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = setup_logger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize Data Loader
    data_loader = DataLoader(cfg.dataset.path)

    # Load Corpus
    logger.info("Loading corpus...")
    corpus = data_loader.load_corpus()

    # Initialize Engine
    engine = DevRAG(OmegaConf.to_container(cfg, resolve=True))
    output_path = os.path.join(cfg.output_dir, "final_results.jsonl")

    existing_results = data_loader.load_results(output_path)
    completed_ids = {
        str(item.get("id"))
        for item in existing_results
        if isinstance(item, dict) and item.get("id") is not None
    }
    if completed_ids:
        logger.info(f"Resuming from existing results: {len(completed_ids)} completed queries found in {output_path}")

    # Build Indices and Run Queries in a single event loop to avoid cross-loop semaphore issues
    async def run_all():
        logger.info("Building indices...")
        await engine.build_indices(corpus)

        # Load Queries
        logger.info("Loading queries...")
        queries = data_loader.load_queries()

        # Configure concurrency for queries
        query_concurrency = cfg.search.get('query_concurrency', 8) if 'search' in cfg and cfg.search is not None else 8
        target_queries = queries
        pending_queries = [q for q in target_queries if str(q.get("id")) not in completed_ids]

        logger.info(
            f"Prepared {len(target_queries)} queries for this run window; "
            f"skipping {len(target_queries) - len(pending_queries)} completed queries, "
            f"running {len(pending_queries)} pending queries."
        )

        sem = asyncio.Semaphore(int(query_concurrency))

        async def run_one(q_item):
            async with sem:
                q_text = q_item.get("text") or q_item.get("query")
                try:
                    res = await engine.search(q_text, request_id=q_item.get("id"))
                    if isinstance(res, dict):
                        res.setdefault("id", q_item.get("id"))
                    else:
                        res = {"result": res, "id": q_item.get("id")}
                    return res
                except Exception as e:
                    logger.error(f"Search failed for query {q_item.get('id')}: {e}")
                    return {"query": q_text, "error": str(e), "id": q_item.get("id")}

        tasks = [asyncio.create_task(run_one(q)) for q in pending_queries]

        new_results = []
        with tqdm(total=len(tasks), desc="Running queries") as pbar:
            for fut in asyncio.as_completed(tasks):
                res = await fut
                new_results.append(res)
                completed_ids.add(str(res.get("id")))
                data_loader.append_result(res, output_path)
                pbar.update(1)

        return existing_results + new_results

    results = asyncio.run(run_all())
    results = sorted(results, key=lambda x: x.get("id", 0))
    data_loader.save_results(results, output_path)
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
