import sys
import os
import json
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import warnings

from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from execution import (
    execute_pandas_query,
    perform_query_on_postgre_databases,
    format_float,
    load_jsonl,
    save_jsonl,
)

warnings.filterwarnings("ignore")


def calculate_accuracy(
    results: List[Dict[str, Any]],
    predict_data: List[Dict[str, Any]],
    source: str,
    target: str,
    model_name: str,
    subfolder: str,
    record_folder: str,
    prompt_version: str,
) -> float:
    """
    Calculate and record accuracy and error statistics from the evaluation results.

    Args:
        results (List[Dict[str, Any]]): Evaluation results for each prediction.
        predict_data (List[Dict[str, Any]]): Prediction entries corresponding to the results.
        source (str): Source database type (e.g., "PostgreSQL" or "Pandas").
        target (str): Target database type.
        model_name (str): Name of the model used for generating predictions.
        subfolder (str): Subfolder or version identifier.
        record_folder (str): Directory where record files are stored.
        prompt_version (str): Prompt version identifier.

    Returns:
        float: The calculated accuracy percentage.
    """
    num_queries = len(results)
    correct_count = sum(1 for res in results if res["passed"])
    execution_errors = sum(
        1 for res in results if res["result"].startswith("Execution failed:")
    )
    assertion_errors = sum(
        1 for res in results if res["result"].startswith("Assertion failed:")
    )
    timeout_errors = sum(1 for res in results if res["result"] == "timed out")
    accuracy = (correct_count / num_queries) * 100

    stats_message = (
        f"Model: {model_name}\n"
        f"Prompt Version: {prompt_version}\n"
        f"{source} to {target} result statistics:\n"
        f"Execution Errors: {execution_errors}\n"
        f"Timeout Errors: {timeout_errors}\n"
        f"Assertion Errors: {assertion_errors}\n"
        f"Total Errors: {num_queries - correct_count}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Timestamp: {datetime.now()}\n"
    )

    print(stats_message)

    # Record results to a file
    record_file_name = f"{source}_to_{target}_{subfolder}.txt"
    record_file_path = os.path.join(record_folder, record_file_name)
    os.makedirs(record_folder, exist_ok=True)

    # If the record file does not exist, create it and add a header
    if not os.path.exists(record_file_path):
        with open(record_file_path, "w", encoding="utf-8") as f:
            f.write("Record file created.\n")
            f.write("-" * 50 + "\n")

    # Append current statistics
    with open(record_file_path, "a", encoding="utf-8") as f:
        f.write(stats_message)
        f.write("-" * 50 + "\n")

    return accuracy


def evaluate_single_prediction(
    gt_query: str, db_path: str, pred_query: str, target: str, table_list: List[str]
) -> Dict[str, Any]:
    """
    Evaluate a single prediction by comparing the predicted query results with the ground truth.

    This function executes the ground truth query and predicted query in different environments
    (depending on `target`), then compares their outputs for correctness.

    Args:
        gt_query (str): Ground truth query.
        db_path (str): Path to the database file or configuration.
        pred_query (str): Predicted query to be evaluated.
        target (str): Target database type ("PostgreSQL" or "Pandas").
        table_list (List[str]): List of tables involved.

    Returns:
        Dict[str, Any]: Dictionary with 'passed' (bool) and 'result' (str) indicating the outcome.
    """
    try:
        pandas_db_path = "/src/pandas_db"
        if target == "PostgreSQL":
            gt_result = execute_pandas_query(pandas_db_path, table_list, gt_query)
            pred_result = perform_query_on_postgre_databases(pred_query)
        else:
            gt_result = perform_query_on_postgre_databases(gt_query)
            pred_result = execute_pandas_query(pandas_db_path, table_list, pred_query)

        # Format results to handle floating point differences
        gt_result = format_float(gt_result)
        pred_result = format_float(pred_result)

        if gt_result == pred_result:
            return {"passed": True, "result": "correct"}
        else:
            return {"passed": False, "result": "Assertion failed: mismatch"}
    except Exception as e:
        return {"passed": False, "result": f"Execution failed: {str(e)}"}


def evaluate_predictions(
    gt_data: List[Dict[str, Any]],
    predict_data: List[Dict[str, Any]],
    source: str,
    target: str,
    timeout: int,
    max_workers: int,
    model_name: str,
    subfolder: str,
    record_folder: str,
    error_jsonl_path: str,
    EA_flag: bool = False,
    prompt_version: str = "v1",
) -> float:
    """
    Evaluate all predictions using multiple threads with a timeout.

    This function:
    - Executes each prediction in parallel with a timeout.
    - Collects errors and optionally saves them to a JSONL file.
    - Calculates and prints the accuracy and other statistics.

    Args:
        gt_data (List[Dict[str, Any]]): Ground truth data instances.
        predict_data (List[Dict[str, Any]]): Corresponding prediction data instances.
        source (str): Source database type.
        target (str): Target database type.
        timeout (int): Timeout in seconds for each evaluation.
        max_workers (int): Number of parallel worker threads.
        model_name (str): Model name for reporting.
        subfolder (str): Subfolder/version identifier.
        record_folder (str): Directory to store result records.
        error_jsonl_path (str): Path to save error cases.
        EA_flag (bool): Whether to write error cases to a JSONL file.
        prompt_version (str): Prompt version identifier.

    Returns:
        float: The accuracy of the predictions.
    """
    error_cases = []
    results = []

    def evaluate_with_timeout(*args, **kwargs):
        """Execute `evaluate_single_prediction` with a timeout."""
        from func_timeout import func_timeout, FunctionTimedOut

        try:
            return func_timeout(timeout, evaluate_single_prediction, args=args)
        except FunctionTimedOut:
            return {"passed": False, "result": "timed out"}

    # Submit evaluation tasks to a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_with_timeout,
                gt["SQL"] if source == "PostgreSQL" else gt["pandas_query"],
                gt["db_path"],
                predict["response"],
                target,
                gt["tables"],
            )
            for gt, predict in zip(gt_data, predict_data)
        ]

        # Process results as they complete
        for idx, future in enumerate(
            tqdm(
                futures,
                total=len(futures),
                desc="Evaluating predictions",
                colour="green",
            )
        ):
            try:
                res = future.result()
                results.append(res)
                error_cases.append(
                    {
                        "model": model_name,
                        "source": source,
                        "target": target,
                        "ground_truth": gt_data[idx],
                        "predicted": predict_data[idx],
                        "error_result": res["result"],
                        "passed": res["passed"],
                    }
                )
            except Exception as e:
                # Handle unexpected errors during evaluation
                error_cases.append(
                    {
                        "model": model_name,
                        "source": source,
                        "target": target,
                        "ground_truth": gt_data[idx],
                        "predicted": predict_data[idx],
                        "error_result": str(e),
                        "passed": False,
                    }
                )

    # Optionally save error cases
    if error_cases and EA_flag:
        save_jsonl(error_cases, error_jsonl_path)

    # Calculate and record accuracy
    return calculate_accuracy(
        results,
        predict_data,
        source,
        target,
        model_name,
        subfolder,
        record_folder,
        prompt_version,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SQL <-> Pandas query transformations."
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=True,
        help="Path to the ground truth JSONL file.",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to the predictions JSONL file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for each evaluation in seconds.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Maximum number of parallel workers."
    )
    parser.add_argument(
        "--record_folder",
        type=str,
        default="record",
        help="Folder to store evaluation records.",
    )
    parser.add_argument(
        "--error_jsonl_path", type=str, required=False, help="Path to save error cases."
    )

    args = parser.parse_args()

    # Derive subfolder, model name, and prompt version from the predictions path
    subfolder = os.path.basename(os.path.dirname(args.predictions_path))
    model_name = os.path.basename(args.predictions_path).split("_")[0]
    prompt_version = os.path.basename(args.predictions_path).split("_")[1]

    # Load ground truth and predictions
    gt_data = load_jsonl(args.ground_truth_path)
    predict_data = load_jsonl(args.predictions_path)

    # Determine when to save error cases
    EA_flag = False
    if (
        "zero_shot" in args.predictions_path
        or "rag" in args.predictions_path
        or "intent" in args.predictions_path
    ):
        EA_flag = True

    # Determine source and target from file naming convention
    if "postgresql_to_pandas" in args.predictions_path:
        source, target = "PostgreSQL", "Pandas"
    else:
        source, target = "Pandas", "PostgreSQL"

    print(f"Evaluating {source} to {target} translations...")

    # Evaluate predictions and print results
    accuracy = evaluate_predictions(
        gt_data,
        predict_data,
        source,
        target,
        args.timeout,
        args.max_workers,
        model_name,
        subfolder,
        args.record_folder,
        args.error_jsonl_path,
        EA_flag,
        prompt_version,
    )
