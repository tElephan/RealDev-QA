import argparse
import asyncio
import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.core.engine import DevRAG
from src.io.data_loader import DataLoader
from src.utils.helpers import load_json, read_jsonl, save_json
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

KS = [1, 5, 10, 20]
DEFAULT_EXPERIMENTS = ["RealDev-QA", "without_planner", "without_verifier", "without_reflector"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DevRAG agentic-module ablations. Planner and verifier are replaced "
            "by controlled stub modules; reflector is ablated by limiting max_turns."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Project root containing the config/ directory. Defaults to the repository root.",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Root config file name inside config/, without the .yaml suffix.",
    )
    parser.add_argument("--dataset", default=None, help="Optional dataset config override.")
    parser.add_argument("--model", default=None, help="Optional model config override.")
    parser.add_argument("--search", default=None, help="Optional search config override.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help=(
            "Built-ins: RealDev-QA, full, without_planner, without_verifier, without_reflector; "
            "aliases: no_planner, no_verifier, no_reflector; reflector budgets: "
            "turns_N, max_turns_N, reflector_turns_N, or max_turns=N."
        ),
    )
    parser.add_argument(
        "--query-ids",
        nargs="+",
        default=None,
        help="Explicit query ids to run. Example: --query-ids 1 5 42",
    )
    parser.add_argument(
        "--query-ids-file",
        default=None,
        help="Optional query subset file. Supports .txt, .json, and .jsonl.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for experiment outputs. Defaults to outputs/ablation/<dataset_name>.",
    )
    parser.add_argument(
        "--query-concurrency",
        type=int,
        default=None,
        help="Override query concurrency. Defaults to search.query_concurrency or 8.",
    )
    parser.add_argument(
        "--reuse-indices-only",
        action="store_true",
        help="Fail fast if any required prebuilt index is missing.",
    )
    return parser.parse_args()


def register_hydra_runtime_resolver(project_root: Path) -> None:
    runtime = {"runtime": {"cwd": str(project_root)}}

    def hydra_resolver(path: str) -> Any:
        value: Any = runtime
        for part in str(path).split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                raise KeyError(f"Unsupported hydra resolver path: {path}")
        return value

    OmegaConf.clear_resolver("hydra")
    OmegaConf.register_new_resolver("hydra", hydra_resolver)


def load_repo_config(
    project_root: Path,
    config_name: str,
    dataset_override: Optional[str] = None,
    model_override: Optional[str] = None,
    search_override: Optional[str] = None,
) -> DictConfig:
    register_hydra_runtime_resolver(project_root)

    config_dir = project_root / "config"
    root_cfg_path = config_dir / f"{config_name}.yaml"
    if not root_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {root_cfg_path}")

    root_cfg = OmegaConf.load(root_cfg_path)
    defaults = list(root_cfg.get("defaults", []))

    chosen_groups: Dict[str, str] = {}
    for item in defaults:
        if isinstance(item, (dict, DictConfig)):
            for group_name, group_value in item.items_ex(resolve=False):
                chosen_groups[group_name] = group_value

    if dataset_override:
        chosen_groups["dataset"] = dataset_override
    if model_override:
        chosen_groups["model"] = model_override
    if search_override:
        chosen_groups["search"] = search_override

    merged = OmegaConf.create({"hydra": {"runtime": {"cwd": str(project_root)}}})

    for group_name, group_value in chosen_groups.items():
        if group_name == "_self_":
            continue
        group_cfg_path = config_dir / group_name / f"{group_value}.yaml"
        if not group_cfg_path.exists():
            raise FileNotFoundError(f"Config group file not found: {group_cfg_path}")
        merged = OmegaConf.merge(
            merged,
            OmegaConf.create({group_name: OmegaConf.load(group_cfg_path)}),
        )

    root_without_defaults = OmegaConf.create(OmegaConf.to_container(root_cfg, resolve=False))
    if "defaults" in root_without_defaults:
        del root_without_defaults["defaults"]

    merged = OmegaConf.merge(merged, root_without_defaults)
    OmegaConf.resolve(merged)
    return merged


def load_query_id_subset(args: argparse.Namespace) -> Optional[Set[str]]:
    ids: Set[str] = set()

    if args.query_ids:
        ids.update(str(qid) for qid in args.query_ids)

    if args.query_ids_file:
        ids.update(load_query_ids_from_file(Path(args.query_ids_file)))

    return ids or None


def load_query_ids_from_file(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Query id file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    if suffix == ".json":
        payload = load_json(str(path))
        if isinstance(payload, list):
            return {str(item["id"] if isinstance(item, dict) and "id" in item else item) for item in payload}
        if isinstance(payload, dict):
            if "query_ids" in payload and isinstance(payload["query_ids"], list):
                return {str(item) for item in payload["query_ids"]}
            if "ids" in payload and isinstance(payload["ids"], list):
                return {str(item) for item in payload["ids"]}
        raise ValueError(f"Unsupported JSON payload in {path}. Expected an array or an object with query_ids/ids.")

    if suffix == ".jsonl":
        ids = set()
        for item in read_jsonl(str(path)):
            if "id" in item:
                ids.add(str(item["id"]))
            elif "query_id" in item:
                ids.add(str(item["query_id"]))
            else:
                raise ValueError(f"Unsupported JSONL row in {path}: {item}")
        return ids

    raise ValueError(f"Unsupported query id file format: {path}")


def select_queries(
    all_queries: Sequence[Dict[str, Any]],
    query_id_subset: Optional[Set[str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not query_id_subset:
        return list(all_queries), []

    query_map = {str(item.get("id")): item for item in all_queries if item.get("id") is not None}
    selected_ids = [qid for qid in query_map.keys() if qid in query_id_subset]
    missing_ids = sorted(query_id_subset - set(selected_ids))
    selected_queries = [query_map[qid] for qid in selected_ids]
    return selected_queries, missing_ids


def ensure_indices_ready(storage_path: str, ontology_enabled: bool = False) -> None:
    required_paths = [
        Path(storage_path) / "bm25",
        Path(storage_path) / "vector",
        Path(storage_path) / "vector.ids.json",
        Path(storage_path) / "entity",
        Path(storage_path) / "entity.appc.json",
    ]
    if ontology_enabled:
        required_paths.append(Path(storage_path) / "ontology.json")

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prebuilt indices required by --reuse-indices-only:\n" + "\n".join(missing)
        )


def build_output_root(project_root: Path, dataset_name: str, output_root_arg: Optional[str]) -> Path:
    if output_root_arg:
        return Path(output_root_arg)
    return project_root / "outputs" / "ablation" / dataset_name


def build_experiment_specs(experiment_names: Iterable[str]) -> List[Dict[str, Any]]:
    return [parse_experiment_spec(raw_name) for raw_name in experiment_names]


def parse_experiment_spec(raw_name: str) -> Dict[str, Any]:
    if raw_name in {"RealDev-QA", "full"}:
        return {
            "name": raw_name,
            "agentic_modules": {},
            "search_overrides": {},
            "source": raw_name,
        }

    if raw_name in {"no_planner", "without_planner"}:
        return {
            "name": "without_planner",
            "agentic_modules": {"planner": False},
            "search_overrides": {},
            "source": raw_name,
        }

    if raw_name in {"no_verifier", "without_verifier"}:
        return {
            "name": "without_verifier",
            "agentic_modules": {"verifier": False},
            "search_overrides": {},
            "source": raw_name,
        }

    if raw_name in {"no_reflector", "without_reflector"}:
        return {
            "name": "without_reflector",
            "agentic_modules": {},
            "search_overrides": {"max_turns": 1},
            "source": raw_name,
        }

    if raw_name.startswith("max_turns="):
        max_turns = int(raw_name.split("=", 1)[1])
        return {
            "name": f"max_turns_{max_turns}",
            "agentic_modules": {},
            "search_overrides": {"max_turns": max_turns},
            "source": raw_name,
        }

    max_turns_match = re.fullmatch(r"(?:turns|max_turns|reflector_turns)_(\d+)", raw_name)
    if max_turns_match:
        max_turns = int(max_turns_match.group(1))
        return {
            "name": raw_name,
            "agentic_modules": {},
            "search_overrides": {"max_turns": max_turns},
            "source": raw_name,
        }

    raise ValueError(f"Unsupported ablation preset: {raw_name}")


def make_experiment_cfg(base_cfg: DictConfig, spec: Dict[str, Any], experiment_dir: Path) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    if "agentic_modules" not in cfg or cfg.agentic_modules is None:
        cfg.agentic_modules = OmegaConf.create({})
    for module_name, enabled in spec.get("agentic_modules", {}).items():
        cfg.agentic_modules[module_name] = bool(enabled)

    for key, value in spec.get("search_overrides", {}).items():
        cfg.search[key] = value

    if "storage" not in cfg or cfg.storage is None:
        cfg.storage = OmegaConf.create({})
    cfg.storage.trace_path = str(experiment_dir / "storage")

    return cfg


def order_results_by_queries(
    results: Sequence[Dict[str, Any]],
    query_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    result_by_id: Dict[str, Dict[str, Any]] = {}
    for item in results:
        if isinstance(item, dict) and item.get("id") is not None:
            result_by_id[str(item["id"])] = item

    ordered = [result_by_id[qid] for qid in query_ids if qid in result_by_id]
    extra_ids = sorted(set(result_by_id.keys()) - set(query_ids))
    ordered.extend(result_by_id[qid] for qid in extra_ids)
    return ordered


async def run_queries_realtime(
    engine: DevRAG,
    queries: Sequence[Dict[str, Any]],
    query_concurrency: int,
    data_loader: DataLoader,
    output_path: Path,
) -> List[Dict[str, Any]]:
    output_path_str = str(output_path)
    existing_results = data_loader.load_results(output_path_str)
    query_ids = [str(item.get("id")) for item in queries if item.get("id") is not None]
    completed_ids = {
        str(item.get("id"))
        for item in existing_results
        if isinstance(item, dict) and item.get("id") is not None
    }
    pending_queries = [query for query in queries if str(query.get("id")) not in completed_ids]

    if completed_ids:
        logger.info(
            "Resuming from %s: %d completed queries found, %d pending.",
            output_path_str,
            len(completed_ids),
            len(pending_queries),
        )
    else:
        logger.info("No completed results found at %s; starting fresh.", output_path_str)

    sem = asyncio.Semaphore(int(query_concurrency))

    async def run_one(q_item: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            qid = str(q_item.get("id"))
            q_text = q_item.get("text") or q_item.get("query") or ""
            try:
                res = await engine.search(q_text, request_id=qid)
                if isinstance(res, dict):
                    res.setdefault("id", qid)
                else:
                    res = {"result": res, "id": qid}
                return res
            except Exception as exc:
                logger.exception("Search failed for query %s", qid)
                return {"query": q_text, "error": str(exc), "id": qid, "doc_ids": []}

    tasks = [asyncio.create_task(run_one(query)) for query in pending_queries]
    new_results: List[Dict[str, Any]] = []
    with tqdm(total=len(tasks), desc="Running queries") as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future
            new_results.append(result)
            data_loader.append_result(result, output_path_str)
            completed_ids.add(str(result.get("id")))
            pbar.update(1)

    combined_results = order_results_by_queries(existing_results + new_results, query_ids)
    data_loader.save_results(combined_results, output_path_str)
    return combined_results


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k.intersection(ground_truth_ids)) / len(ground_truth_ids)


def cer_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
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
    relevant_count = len(ground_truth_ids)
    if relevant_count == 0:
        return 0.0
    cutoff = min(relevant_count, k)
    if cutoff <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:cutoff]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / cutoff


def load_relevance(path: str) -> Dict[str, Set[str]]:
    relevance: Dict[str, Set[str]] = {}
    for item in read_jsonl(path):
        query_id = item.get("query-id")
        corpus_id = item.get("corpus-id")
        if query_id is not None and corpus_id is not None:
            relevance.setdefault(str(query_id), set()).add(str(corpus_id))
    return relevance


def load_query_map(path: str) -> Dict[str, str]:
    query_map: Dict[str, str] = {}
    for item in read_jsonl(path):
        query_id = item.get("id")
        text = item.get("text") or item.get("query") or item.get("question") or ""
        if query_id is not None:
            query_map[str(query_id)] = text
    return query_map


def evaluate_results(
    dataset_dir: str,
    results: Sequence[Dict[str, Any]],
    query_ids: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    relevance_path = os.path.join(dataset_dir, "relevance.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")

    rel_map = load_relevance(relevance_path)
    query_map = load_query_map(queries_path)
    res_map = {str(item.get("id")): [str(doc_id) for doc_id in item.get("doc_ids", [])] for item in results}

    metrics_rows: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {
        "recall_all": 0.0,
        "r_precision_all": 0.0,
        "f1_all": 0.0,
        "cer_at_10": 0.0,
    }
    for k in KS:
        totals[f"recall_at_{k}"] = 0.0
        totals[f"r_precision_at_{k}"] = 0.0

    evaluated = 0
    requested_qids = [str(qid) for qid in query_ids]
    missing_from_relevance: List[str] = []
    missing_from_results: List[str] = []

    for query_id in requested_qids:
        if query_id not in rel_map:
            missing_from_relevance.append(query_id)
            continue
        if query_id not in res_map:
            missing_from_results.append(query_id)
            continue

        gt_ids = rel_map[query_id]
        retrieved_ids = res_map[query_id]
        query_text = query_map.get(query_id, "")

        all_k = max(len(retrieved_ids), len(gt_ids))
        recall_all = recall_at_k(retrieved_ids, gt_ids, k=all_k)
        precision_all = precision_at_k(retrieved_ids, gt_ids, k=len(retrieved_ids))
        f1_all = f1_from_precision_recall(precision_all, recall_all)
        r_precision_all = r_precision_at_k(retrieved_ids, gt_ids, k=all_k)

        row: Dict[str, Any] = {
            "query_id": query_id,
            "query_text": query_text,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(gt_ids),
            "cer_at_10": cer_at_k(retrieved_ids, gt_ids, k=10),
            "recall_all": recall_all,
            "r_precision_all": r_precision_all,
            "f1_all": f1_all,
            "missing_doc_ids": ";".join(sorted(gt_ids - set(retrieved_ids))),
        }

        for k in KS:
            row[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gt_ids, k=k)
            row[f"r_precision_at_{k}"] = r_precision_at_k(retrieved_ids, gt_ids, k=k)
            totals[f"recall_at_{k}"] += row[f"recall_at_{k}"]
            totals[f"r_precision_at_{k}"] += row[f"r_precision_at_{k}"]

        totals["recall_all"] += recall_all
        totals["r_precision_all"] += r_precision_all
        totals["f1_all"] += f1_all
        totals["cer_at_10"] += row["cer_at_10"]
        metrics_rows.append(row)
        evaluated += 1

    summary: Dict[str, Any] = {
        "requested_queries": len(requested_qids),
        "evaluated_queries": evaluated,
        "missing_from_relevance": missing_from_relevance,
        "missing_from_results": missing_from_results,
    }

    if evaluated == 0:
        for k in KS:
            summary[f"recall_at_{k}"] = 0.0
            summary[f"r_precision_at_{k}"] = 0.0
        summary["recall_all"] = 0.0
        summary["r_precision_all"] = 0.0
        summary["f1_all"] = 0.0
        summary["cer_at_10"] = 0.0
        return summary, metrics_rows

    for key, value in totals.items():
        summary[key] = value / evaluated

    return summary, metrics_rows


def write_metrics_csv(path: Path, metrics_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_text",
        "cer_at_10",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_all",
        "r_precision_at_1",
        "r_precision_at_5",
        "r_precision_at_10",
        "r_precision_at_20",
        "r_precision_all",
        "f1_all",
        "num_retrieved",
        "num_relevant",
        "missing_doc_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def write_summary_csv(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "requested_queries",
        "evaluated_queries",
        "cer_at_10",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_all",
        "r_precision_at_1",
        "r_precision_at_5",
        "r_precision_at_10",
        "r_precision_at_20",
        "r_precision_all",
        "f1_all",
        "planner_enabled",
        "verifier_enabled",
        "max_turns",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def summarize_row(experiment_name: str, experiment_cfg: DictConfig, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": experiment_name,
        "requested_queries": summary.get("requested_queries", 0),
        "evaluated_queries": summary.get("evaluated_queries", 0),
        "cer_at_10": summary.get("cer_at_10", 0.0),
        "recall_at_1": summary.get("recall_at_1", 0.0),
        "recall_at_5": summary.get("recall_at_5", 0.0),
        "recall_at_10": summary.get("recall_at_10", 0.0),
        "recall_at_20": summary.get("recall_at_20", 0.0),
        "recall_all": summary.get("recall_all", 0.0),
        "r_precision_at_1": summary.get("r_precision_at_1", 0.0),
        "r_precision_at_5": summary.get("r_precision_at_5", 0.0),
        "r_precision_at_10": summary.get("r_precision_at_10", 0.0),
        "r_precision_at_20": summary.get("r_precision_at_20", 0.0),
        "r_precision_all": summary.get("r_precision_all", 0.0),
        "f1_all": summary.get("f1_all", 0.0),
        "planner_enabled": bool(experiment_cfg.agentic_modules.get("planner", True)),
        "verifier_enabled": bool(experiment_cfg.agentic_modules.get("verifier", True)),
        "max_turns": int(experiment_cfg.search.get("max_turns", 0)),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    cfg = load_repo_config(
        project_root=project_root,
        config_name=args.config_name,
        dataset_override=args.dataset,
        model_override=args.model,
        search_override=args.search,
    )
    experiment_specs = build_experiment_specs(args.experiments)

    dataset_name = str(cfg.dataset.name)
    dataset_dir = str(cfg.dataset.path)
    output_root = build_output_root(project_root, dataset_name, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.reuse_indices_only:
        ensure_indices_ready(
            str(cfg.storage.path),
            ontology_enabled=bool(cfg.search.get("ontology_enabled", False)),
        )

    logger.info("Dataset: %s", dataset_name)
    logger.info("Dataset path: %s", dataset_dir)
    logger.info("Storage path: %s", cfg.storage.path)
    logger.info("Experiments: %s", ", ".join(spec["name"] for spec in experiment_specs))

    data_loader = DataLoader(dataset_dir)
    corpus = data_loader.load_corpus()
    all_queries = data_loader.load_queries()
    query_id_subset = load_query_id_subset(args)
    selected_queries, missing_query_ids = select_queries(all_queries, query_id_subset)

    if not selected_queries:
        raise ValueError("No queries selected for the ablation experiment.")

    if missing_query_ids:
        logger.warning("The following query ids were not found and will be skipped: %s", missing_query_ids)

    selected_query_ids = [str(item["id"]) for item in selected_queries]
    query_concurrency = int(args.query_concurrency or cfg.search.get("query_concurrency", 8))

    run_manifest = {
        "dataset": dataset_name,
        "dataset_path": dataset_dir,
        "storage_path": str(cfg.storage.path),
        "selected_query_count": len(selected_queries),
        "selected_query_ids": selected_query_ids,
        "missing_query_ids": missing_query_ids,
        "experiments": experiment_specs,
        "query_concurrency": query_concurrency,
        "reuse_indices_only": bool(args.reuse_indices_only),
    }
    save_json(str(output_root / "run_manifest.json"), run_manifest)

    summary_rows: List[Dict[str, Any]] = []

    for index, spec in enumerate(experiment_specs, start=1):
        experiment_name = spec["name"]
        experiment_dir = output_root / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%d/%d] Running experiment '%s'", index, len(experiment_specs), experiment_name)

        experiment_cfg = make_experiment_cfg(base_cfg=cfg, spec=spec, experiment_dir=experiment_dir)
        save_json(
            str(experiment_dir / "experiment_manifest.json"),
            {
                "experiment": experiment_name,
                "agentic_modules": OmegaConf.to_container(experiment_cfg.agentic_modules, resolve=True),
                "search_overrides": spec.get("search_overrides", {}),
                "max_turns": int(experiment_cfg.search.get("max_turns", 0)),
                "storage_path": str(experiment_cfg.storage.path),
                "trace_path": str(experiment_cfg.storage.trace_path),
                "source": spec.get("source"),
                "query_ids": selected_query_ids,
            },
        )

        engine = DevRAG(OmegaConf.to_container(experiment_cfg, resolve=True))
        results_path = experiment_dir / "final_results.jsonl"

        async def _run_experiment() -> List[Dict[str, Any]]:
            await engine.build_indices(corpus)
            return await run_queries_realtime(
                engine=engine,
                queries=selected_queries,
                query_concurrency=query_concurrency,
                data_loader=data_loader,
                output_path=results_path,
            )

        results = asyncio.run(_run_experiment())

        summary, metrics_rows = evaluate_results(
            dataset_dir=dataset_dir,
            results=results,
            query_ids=selected_query_ids,
        )
        save_json(str(experiment_dir / "evaluation_summary.json"), summary)
        write_metrics_csv(experiment_dir / "evaluation_metrics.csv", metrics_rows)

        summary_row = summarize_row(experiment_name, experiment_cfg, summary)
        summary_rows.append(summary_row)

        logger.info(
            "Experiment '%s' finished. Recall@10=%.4f, Recall@All=%.4f, F1@All=%.4f",
            experiment_name,
            summary_row["recall_at_10"],
            summary_row["recall_all"],
            summary_row["f1_all"],
        )

    write_summary_csv(output_root / "ablation_summary.csv", summary_rows)
    save_json(str(output_root / "ablation_summary.json"), summary_rows)

    logger.info("Ablation experiments completed. Summary saved to %s", output_root / "ablation_summary.csv")


if __name__ == "__main__":
    main()
