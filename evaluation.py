import asyncio
import csv
import glob
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.core.resources import ResourceManager
from src.models.remote_model import RemoteLLMClient
from src.utils.helpers import read_jsonl, save_json
from src.utils.logger import setup_logger

# Silence noisy per-request transport logs during evaluation.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = setup_logger(__name__)

KS = [1, 5, 10, 20]
LLM_ACC_PROMPT_TEMPLATE = """
You are an evaluator.
Judge only whether the predicted answer identifies the same concrete fix method as the gold answer.
Do not reward length, citations, markdown, tables, or extra background.
Return "1" if the predicted answer uses the same fix method as the gold answer, even if phrased differently.
Return "0" if it uses a different fix, gives only a generic workaround, or is too vague to identify the fix method.
When in doubt, return "0".

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {answer}

Return only "1" if correct or "0" if incorrect:


""".strip()

def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k.intersection(ground_truth_ids)) / len(ground_truth_ids)


def cer_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Complete Evidence Recall@k: whether all evidence docs are within top-k."""
    if not ground_truth_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return 1.0 if ground_truth_ids.issubset(retrieved_k) else 0.0

def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / len(retrieved_k)

def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def r_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Truncated R-Precision at k:
    precision over the first min(R, k) results, where R = number of relevant docs.
    This keeps the metric comparable across fixed cutoffs and @all.
    """
    R = len(ground_truth_ids)
    if R == 0:
        return 0.0
    cutoff = min(R, k)
    if cutoff <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:cutoff]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / cutoff


def resolve_dataset_file(dataset_dir: str, preferred_name: str, fallback_glob: Optional[str] = None) -> str:
    preferred_path = os.path.join(dataset_dir, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path

    if fallback_glob:
        matches = sorted(glob.glob(os.path.join(dataset_dir, fallback_glob)))
        if matches:
            return matches[0]

    return preferred_path


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, DictConfig):
        return config.get(key, default)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def normalize_model_suffix(value: str) -> str:
    model_name = str(value).strip().lower()
    provider_match = re.match(r"[a-z]+", model_name)
    suffix = provider_match.group(0) if provider_match else model_name
    suffix = re.sub(r"[^a-z0-9]+", "_", suffix).strip("_")
    return suffix or "model"


def model_suffix_from_cfg(cfg: DictConfig) -> str:
    """Build a short, filesystem-safe suffix from the evaluation LLM name."""
    eval_cfg = cfg_get(cfg, "evaluation", None)
    result_suffix = cfg_get(eval_cfg, "result_suffix", None)
    if result_suffix:
        return normalize_model_suffix(result_suffix)

    model_name = cfg_get(eval_cfg, "llm_name", cfg.model.get("llm_name", "model"))
    return normalize_model_suffix(model_name)


def suffixed_output_path(output_dir: str, filename: str, suffix: str) -> str:
    stem, ext = os.path.splitext(filename)
    return os.path.join(output_dir, f"{stem}_{suffix}{ext}")


def load_relevance(path: str) -> Dict[str, Set[str]]:
    """Load relevance judgments: query_id -> set of relevant doc_ids"""
    relevance = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("query-id")
        docid = item.get("corpus-id")
        if qid is not None and docid is not None:
            qid = str(qid)
            if qid not in relevance:
                relevance[qid] = set()
            relevance[qid].add(str(docid))
    return relevance

def load_queries(path: str) -> Dict[str, str]:
    """Load queries: query_id -> query_text"""
    queries = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        text = item.get("text") or item.get("query") or item.get("question") or ""
        if qid is not None:
            queries[str(qid)] = text
    return queries


def load_gold_answers(path: str) -> Dict[str, str]:
    """Load gold final answers: query_id -> gold_answer"""
    gold_answers = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        answer = item.get("final_answer") or item.get("answer") or ""
        if qid is not None:
            gold_answers[str(qid)] = answer
    return gold_answers

def load_results(path: str) -> Dict[str, List[str]]:
    """Load system results: query_id -> list of retrieved doc_ids"""
    results = {}
    if not os.path.exists(path):
        print(f"Warning: Results file not found at {path}")
        return results
        
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        # Support both "doc_ids" (list) or infer from other fields if needed
        doc_ids = item.get("doc_ids", [])
        if qid is not None:
            # Ensure doc_ids are strings
            results[str(qid)] = [str(d) for d in doc_ids]
    return results


def load_result_items(path: str) -> Dict[str, Dict[str, Any]]:
    """Load raw result rows keyed by query id."""
    results = {}
    if not os.path.exists(path):
        return results

    for item in read_jsonl(path):
        qid = item.get("id")
        if qid is not None:
            results[str(qid)] = item
    return results


def parse_binary_judgment(raw_output: str) -> Optional[int]:
    """Parse the judge model output into 0/1."""
    text = (raw_output or "").strip()
    if text in {"0", "1"}:
        return int(text)

    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))

    collapsed = "".join(ch for ch in text if ch in {"0", "1"})
    if len(collapsed) == 1:
        return int(collapsed)

    return None


async def evaluate_llm_acc(
    cfg: DictConfig,
    dataset_dir: str,
    results_path: str,
    output_dir: str,
    model_suffix: str,
) -> Optional[Dict[str, Any]]:
    queries_path = resolve_dataset_file(dataset_dir, "queries.jsonl", "queries*.jsonl")
    gold_answers_path = resolve_dataset_file(dataset_dir, "final_answer.jsonl", "final_answer*.jsonl")

    if not os.path.exists(gold_answers_path):
        logger.warning("Gold answers file not found at %s; skipping llm-acc evaluation.", gold_answers_path)
        return None

    result_items = load_result_items(results_path)
    if not result_items:
        logger.warning("No final answers found in %s; skipping llm-acc evaluation.", results_path)
        return None

    query_map = load_queries(queries_path) if os.path.exists(queries_path) else {}
    gold_map = load_gold_answers(gold_answers_path)

    eval_qids = sorted(qid for qid in gold_map.keys() if qid in result_items)
    missing_predictions = sorted(qid for qid in gold_map.keys() if qid not in result_items)

    if not eval_qids:
        logger.warning("No overlapping ids between gold answers and predictions; skipping llm-acc evaluation.")
        return None

    eval_cfg = cfg_get(cfg, "evaluation", None)
    judge_model = cfg_get(eval_cfg, "llm_name", cfg.model.llm_name)
    judge_api_key = cfg_get(eval_cfg, "api_key", cfg.model.api_key)
    judge_base_url = cfg_get(eval_cfg, "base_url", cfg.model.base_url)
    judge_temperature = float(cfg_get(eval_cfg, "temperature", 0))
    judge_max_tokens = int(cfg_get(eval_cfg, "max_tokens", 8))
    judge_max_retries = int(cfg_get(eval_cfg, "max_retries", 3))

    ResourceManager.setup(OmegaConf.to_container(cfg.resources, resolve=True))
    judge_client = RemoteLLMClient(
        api_key=judge_api_key,
        base_url=judge_base_url,
        model=judge_model,
    )

    llm_judge_concurrency = cfg_get(eval_cfg, "llm_judge_concurrency", None)
    if llm_judge_concurrency is None:
        llm_judge_concurrency = cfg_get(eval_cfg, "concurrency", None)
    if llm_judge_concurrency is None:
        llm_judge_concurrency = cfg_get(cfg_get(cfg, "search", None), "query_concurrency", 8)
    llm_judge_concurrency = int(llm_judge_concurrency)

    sem = asyncio.Semaphore(max(1, llm_judge_concurrency))
    logger.info(
        "Running llm-acc evaluation for %s queries with judge model=%s concurrency=%s temperature=%s.",
        len(eval_qids),
        judge_model,
        llm_judge_concurrency,
        judge_temperature,
    )

    async def judge_one(qid: str) -> Dict[str, Any]:
        result_item = result_items[qid]
        question = str(query_map.get(qid) or result_item.get("query") or result_item.get("text") or "")
        gold_answer = str(gold_map[qid])
        predicted_answer = str(
            result_item.get("answer")
            or result_item.get("final_answer")
            or result_item.get("result")
            or ""
        )
        prompt = LLM_ACC_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            answer=predicted_answer,
        )

        async with sem:
            raw_output = await judge_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=judge_temperature,
                max_tokens=judge_max_tokens,
                max_retries=judge_max_retries,
            )

        label = parse_binary_judgment(raw_output)
        return {
            "query_id": qid,
            "query_text": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "judge_raw_output": raw_output,
            "llm_label": label,
        }

    async def judge_one_with_qid(qid: str):
        return qid, await judge_one(qid)

    tasks = [asyncio.create_task(judge_one_with_qid(qid)) for qid in eval_qids]
    rows_by_qid: Dict[str, Dict[str, Any]] = {}
    with tqdm(total=len(tasks), desc="LLM-Acc judging", unit="query") as pbar:
        for task in asyncio.as_completed(tasks):
            qid, row = await task
            rows_by_qid[qid] = row
            pbar.update(1)

    rows = [rows_by_qid[qid] for qid in eval_qids]

    valid_rows = [row for row in rows if row["llm_label"] in {0, 1}]
    invalid_rows = [row for row in rows if row["llm_label"] not in {0, 1}]
    llm_acc = (
        sum(int(row["llm_label"]) for row in valid_rows) / len(valid_rows)
        if valid_rows
        else 0.0
    )

    details_csv_path = suffixed_output_path(output_dir, "llm_acc_details.csv", model_suffix)
    with open(details_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id",
            "query_text",
            "llm_label",
            "judge_raw_output",
            "gold_answer",
            "predicted_answer",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    judgments_json_path = suffixed_output_path(output_dir, "llm_acc_judgments.json", model_suffix)
    save_json(
        judgments_json_path,
        {
            row["query_id"]: {
                "query_text": row["query_text"],
                "gold_answer": row["gold_answer"],
                "predicted_answer": row["predicted_answer"],
                "judge_raw_output": row["judge_raw_output"],
                "llm_label": row["llm_label"],
            }
            for row in rows
        },
    )

    summary = {
        "requested_gold_answers": len(gold_map),
        "evaluated_queries": len(rows),
        "valid_judgments": len(valid_rows),
        "invalid_judgments": len(invalid_rows),
        "missing_predictions": missing_predictions,
        "llm_acc": llm_acc,
        "prompt_template": LLM_ACC_PROMPT_TEMPLATE,
        "judge_model": judge_model,
        "model_suffix": model_suffix,
        "judge_temperature": judge_temperature,
        "judge_max_tokens": judge_max_tokens,
        "judge_max_retries": judge_max_retries,
        "judge_concurrency": llm_judge_concurrency,
    }
    return {
        "summary": summary,
        "rows": rows,
        "details_csv_path": details_csv_path,
        "judgments_json_path": judgments_json_path,
    }

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Evaluating with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Paths
    eval_cfg = cfg_get(cfg, "evaluation", None)
    eval_dataset_cfg = cfg_get(eval_cfg, "dataset", None)
    dataset_name = cfg_get(eval_dataset_cfg, "name", cfg.dataset.name)
    dataset_dir = cfg_get(eval_dataset_cfg, "path", cfg.dataset.path)
    output_dir = cfg_get(eval_cfg, "output_dir", None)
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(str(cfg.output_dir)), str(dataset_name))
    output_dir = str(output_dir)
    model_suffix = model_suffix_from_cfg(cfg)
    
    relevance_path = os.path.join(dataset_dir, "relevance.jsonl")
    queries_path = resolve_dataset_file(dataset_dir, "queries.jsonl", "queries*.jsonl")
    results_path = os.path.join(output_dir, "final_results.jsonl")
    summary_output_path = suffixed_output_path(output_dir, "evaluation_summary.json", model_suffix)

    logger.info(f"Loading data for dataset: {dataset_name}")
    rel_map = load_relevance(relevance_path)
    query_map = load_queries(queries_path)
    res_map = load_results(results_path)

    if not res_map:
        logger.warning("No retrieval results found for recall-style metrics.")

    logger.info(f"Found {len(rel_map)} queries in ground truth.")
    logger.info(f"Found {len(res_map)} queries in results.")

    metrics_rows = []
    totals = {
        "recall_all": 0.0,
        "r_precision_all": 0.0,
        "f1_all": 0.0,
        "cer_at_10": 0.0,
    }
    for k in KS:
        totals[f"recall_at_{k}"] = 0.0
        totals[f"r_precision_at_{k}"] = 0.0
    count = 0

    # Evaluate intersection of queries found in both (or just results)
    # Usually we evaluate on all queries in ground truth, treating missing results as 0
    
    eval_qids = sorted(list(rel_map.keys()))
    
    for qid in eval_qids:
        if qid not in res_map:
            # If query was not processed, skip or count as 0? 
            # Usually count as 0, but here we might have partial runs.
            # Let's track it but maybe not penalize if we just ran a subset.
            continue

        gt_ids = rel_map[qid]
        retrieved_ids = res_map[qid]
        query_text = query_map.get(qid, "")

        all_k = max(len(retrieved_ids), len(gt_ids))
        recall_all = recall_at_k(retrieved_ids, gt_ids, k=all_k)
        precision_all = precision_at_k(retrieved_ids, gt_ids, k=len(retrieved_ids))
        f1_all = f1_from_precision_recall(precision_all, recall_all)
        r_precision_all = r_precision_at_k(retrieved_ids, gt_ids, k=all_k)

        row = {
            "query_id": qid,
            "query_text": query_text,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(gt_ids),
            "cer_at_10": cer_at_k(retrieved_ids, gt_ids, k=10),
        }

        for k in KS:
            row[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gt_ids, k=k)
            row[f"r_precision_at_{k}"] = r_precision_at_k(retrieved_ids, gt_ids, k=k)

        row["recall_all"] = recall_all
        row["r_precision_all"] = r_precision_all
        row["f1_all"] = f1_all

        # Missing Docs
        retrieved_set = set(retrieved_ids)
        missing_ids = list(gt_ids - retrieved_set)
        row["missing_doc_ids"] = ";".join(missing_ids)
        metrics_rows.append(row)

        for k in KS:
            totals[f"recall_at_{k}"] += row[f"recall_at_{k}"]
            totals[f"r_precision_at_{k}"] += row[f"r_precision_at_{k}"]
        totals["recall_all"] += recall_all
        totals["r_precision_all"] += r_precision_all
        totals["f1_all"] += f1_all
        totals["cer_at_10"] += row["cer_at_10"]
        count += 1

    retrieval_summary: Dict[str, Any] = {
        "requested_queries": len(eval_qids),
        "evaluated_queries": count,
    }

    if count == 0:
        logger.warning("No matching queries found for retrieval metrics; skipping recall-style evaluation output.")
    else:
        avg_metrics = {name: value / count for name, value in totals.items()}
        retrieval_summary.update(avg_metrics)

        print("-" * 40)
        print(f"Evaluation Results ({count} queries):")
        for k in KS:
            print(f"Average Recall@{k}:       {avg_metrics[f'recall_at_{k}']:.4f}")
        print(f"Average Recall@All:     {avg_metrics['recall_all']:.4f}")
        print(f"Average CER@10:         {avg_metrics['cer_at_10']:.4f}")
        for k in KS:
            print(f"Average R-Precision@{k}: {avg_metrics[f'r_precision_at_{k}']:.4f}")
        print(f"Average R-Precision@All:{avg_metrics['r_precision_all']:.4f}")
        print(f"Average F1@All:         {avg_metrics['f1_all']:.4f}")
        print("-" * 40)
    
    if count == 0:
        for k in KS:
            retrieval_summary[f"recall_at_{k}"] = 0.0
            retrieval_summary[f"r_precision_at_{k}"] = 0.0
        retrieval_summary["recall_all"] = 0.0
        retrieval_summary["r_precision_all"] = 0.0
        retrieval_summary["f1_all"] = 0.0
        retrieval_summary["cer_at_10"] = 0.0

    llm_acc_result = asyncio.run(
        evaluate_llm_acc(
            cfg=cfg,
            dataset_dir=dataset_dir,
            results_path=results_path,
            output_dir=output_dir,
            model_suffix=model_suffix,
        )
    )
    if llm_acc_result is not None:
        summary = llm_acc_result["summary"]
        print(f"Average LLM-Acc:       {summary['llm_acc']:.4f}")
        print(f"Valid LLM Judgments:   {summary['valid_judgments']}/{summary['evaluated_queries']}")
        print(f"LLM-Acc details saved to {llm_acc_result['details_csv_path']}")
        print(f"LLM-Acc judgments saved to {llm_acc_result['judgments_json_path']}")

    evaluation_summary = {
        "dataset": str(dataset_name),
        "dataset_dir": dataset_dir,
        "results_path": results_path,
        "output_dir": output_dir,
        "model_suffix": model_suffix,
        "retrieval": retrieval_summary,
        "llm_acc": llm_acc_result["summary"] if llm_acc_result is not None else None,
    }
    save_json(summary_output_path, evaluation_summary)
    print(f"Evaluation summary saved to {summary_output_path}")

if __name__ == "__main__":
    main()
