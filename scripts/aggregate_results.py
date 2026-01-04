#!/usr/bin/env python3
"""Aggregate results from all experiments."""
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging(level=logging.INFO)
    
    logger.info("Aggregating results...")
    
    models_dir = Path("outputs/models")
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "experiments": [],
        "by_method": defaultdict(list),
        "by_dataset": defaultdict(list),
        "by_task_type": defaultdict(list),
    }
    
    # Find all eval_results.json files
    for eval_file in models_dir.rglob("eval_results.json"):
        try:
            with open(eval_file) as f:
                eval_data = json. load(f)
            
            # Extract info
            config = eval_data.get("config", {})
            metrics = eval_data. get("metrics", {})
            by_task = eval_data.get("by_task_type", {})
            
            experiment = {
                "dataset": config.get("dataset", "unknown"),
                "model":  config.get("model", "unknown"),
                "method": config.get("method", "unknown"),
                "seed": config. get("seed", 0),
                "auroc": metrics.get("auroc", 0),
                "auprc": metrics.get("auprc", 0),
                "f1": metrics.get("f1", 0),
                "n_samples": eval_data.get("n_samples", 0),
                "by_task_type":  by_task,
                "path": str(eval_file),
            }
            
            all_results["experiments"]. append(experiment)
            all_results["by_method"][experiment["method"]].append(experiment)
            all_results["by_dataset"][experiment["dataset"]].append(experiment)
            
            # Aggregate by task type
            for task, task_metrics in by_task. items():
                all_results["by_task_type"][task].append({
                    "method": experiment["method"],
                    "model": experiment["model"],
                    **task_metrics
                })
            
        except Exception as e: 
            logger.warning(f"Failed to load {eval_file}:  {e}")
    
    # Compute summary statistics
    summary = {
        "total_experiments": len(all_results["experiments"]),
        "methods": {},
        "task_types": {},
    }
    
    for method, exps in all_results["by_method"]. items():
        if exps:
            aurocs = [e["auroc"] for e in exps]
            summary["methods"][method] = {
                "mean_auroc": sum(aurocs) / len(aurocs),
                "n_experiments": len(exps),
            }
    
    for task, results in all_results["by_task_type"].items():
        if results:
            aurocs = [r["auroc"] for r in results if "auroc" in r]
            if aurocs:
                summary["task_types"][task] = {
                    "mean_auroc": sum(aurocs) / len(aurocs),
                    "n_experiments": len(aurocs),
                }
    
    all_results["summary"] = summary
    
    # Save
    output_path = results_dir / "summary.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=list)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Total experiments:  {summary['total_experiments']}")
    
    print("\nBy Method:")
    print("-" * 40)
    for method, stats in summary["methods"].items():
        print(f"  {method}: Mean AUROC = {stats['mean_auroc']:.4f} (n={stats['n_experiments']})")
    
    print("\nBy Task Type:")
    print("-" * 40)
    for task, stats in summary["task_types"].items():
        print(f"  {task}:  Mean AUROC = {stats['mean_auroc']:.4f} (n={stats['n_experiments']})")
    
    print("=" * 80)


if __name__ == "__main__": 
    main()