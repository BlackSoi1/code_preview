import sys
import os
import json
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import warnings
import datetime

from tqdm import tqdm
from execution import check_correctness

warnings.filterwarnings("ignore")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a line of JSON data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Append data to a JSONL file. Each dictionary in `data` is written as a JSON object on a new line.

    Args:
        data (List[Dict[str, Any]]): Data to append to the JSONL file.
        file_path (str): Path to the output JSONL file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def calculate_accuracy(
    results: List[Dict[str, Any]],
    predict_data: List[Dict[str, Any]],
    source: str,
    target: str,
    model_name: str,
    subfolder: str,
    record_folder: str,
) -> float:
    """
    Calculate and record the accuracy and error statistics for the evaluated predictions.

    Args:
        results (List[Dict[str, Any]]): Evaluation results for each prediction.
        predict_data (List[Dict[str, Any]]): Original prediction entries corresponding to these results.
        source (str): Source framework/library name (e.g., "numpy" or "pandas").
        target (str): Target framework/library name.
        model_name (str): Name of the model used.
        subfolder (str): Subfolder/version identifier of the prompt.
        record_folder (str): Directory to store evaluation records.

    Returns:
        float: The accuracy of the predictions.
    """
    num_queries = len(results)
    correct_count = sum(res["passed"] for res in results)
    execution_errors = sum(res["result"].startswith("failed:") for res in results)
    assert_errors = sum(
        res["result"].startswith("Assertion failed:") for res in results
    )
    timeout_errors = sum(res["result"] == "timed out" for res in results)
    accuracy = (correct_count / num_queries) * 100

    stats_message = (
        f"Model: {model_name}\n"
        f"\n{source} To {target} result statistics:\n"
        f"Number of Execution Errors: {execution_errors}\n"
        f"Number of Timeouts: {timeout_errors}\n"
        f"Number of Assertion Errors: {assert_errors}\n"
        f"Total Errors: {num_queries - correct_count}\n"
        f"{source} To {target} Accuracy: {accuracy:.2f}%\n"
        f"Timestamp: {datetime.datetime.now()}\n"
    )

    print(stats_message)

    # Determine record file name based on source-target transformation
    if "numpy" in source:
        record_file_name = f"np_to_pd_{subfolder}.txt"
    else:
        record_file_name = f"pd_to_np_{subfolder}.txt"

    record_file_path = os.path.join(record_folder, record_file_name)
    os.makedirs(record_folder, exist_ok=True)

    # Create record file if not existing
    if not os.path.exists(record_file_path):
        with open(record_file_path, "w", encoding="utf-8") as f:
            f.write("Record file created.\n")
            f.write("-" * 50 + "\n")

    # Append stats to the record file
    try:
        with open(record_file_path, "a", encoding="utf-8") as f:
            f.write(stats_message)
            f.write("-" * 50 + "\n")
    except IOError as e:
        print(f"Error writing to file {record_file_path}: {e}")

    return accuracy


def format_test_program(target_test: str, start_code: str, sol_code: str) -> str:
    """
    Construct the test program by combining start code, test code, and solution code.

    Args:
        target_test (str): The test code snippet (includes test functions).
        start_code (str): Initial code or context.
        sol_code (str): The solution code snippet to be tested.

    Returns:
        str: The formatted test program as a string.
    """
    program = (
        f"{start_code}\n"
        f"{target_test}\n"
        f"code = {repr(sol_code)}\n"
        "test_execution(code)\n"
    )
    if "test_string(" in target_test:
        program += "test_string(code)\n"
    else:
        program += "\n"
    return program


def evaluate_single_prediction(
    target_test: str, start_code: str, pred_code: str, timeout: int
) -> Dict[str, Any]:
    """
    Evaluate a single prediction by running the test program through `check_correctness`.

    Args:
        target_test (str): Test code snippet for the target environment.
        start_code (str): Initial code snippet.
        pred_code (str): Predicted solution code.
        timeout (int): Execution timeout in seconds.

    Returns:
        Dict[str, Any]: The result dictionary with "passed" (bool) and "result" (str).
    """
    program = format_test_program(target_test, start_code, pred_code)
    return check_correctness(program, timeout)


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
) -> float:
    """
    Evaluate multiple predictions in parallel and record error cases.

    Args:
        gt_data (List[Dict[str, Any]]): Ground truth data.
        predict_data (List[Dict[str, Any]]): Predictions to be evaluated.
        source (str): Source library name.
        target (str): Target library name.
        timeout (int): Timeout for each evaluation in seconds.
        max_workers (int): Number of parallel worker threads.
        model_name (str): Name of the model.
        subfolder (str): Version or prompt subfolder name.
        record_folder (str): Directory to store result records.
        error_jsonl_path (str): Path to store error cases in JSONL.
        EA_flag (bool): If True, error cases are recorded.

    Returns:
        float: Computed accuracy of the predictions.
    """
    error_cases = []
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_prediction,
                gt[f"{target}_test_code"],
                gt[f"{target}_start_code"],
                predict["response"]
                .replace("```", "")
                .replace("Python", "")
                .replace("python", "")
                .strip(),
                timeout,
            )
            for gt, predict in zip(gt_data, predict_data)
        ]

        for idx, future in enumerate(
            tqdm(
                futures,
                total=len(futures),
                desc="Evaluating predictions",
                colour="green",
            )
        ):
            res = future.result()
            results.append(res)
            error_case = {
                "model": model_name,
                "source": source,
                "target": target,
                "ground_truth": gt_data[idx],
                "error_result": res["result"],
                "prediction": predict_data[idx],
                "passed": res["passed"],
            }
            error_cases.append(error_case)

    # Optionally write error cases
    if error_cases and EA_flag:
        save_jsonl(error_cases, error_jsonl_path)

    return calculate_accuracy(
        results, predict_data, source, target, model_name, subfolder, record_folder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate code conversion.")
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=True,
        help="Path to the ground truth JSONL file",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to the predictions JSONL file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout (in seconds) for each prediction evaluation",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of parallel worker threads",
    )
    parser.add_argument(
        "--record_folder",
        type=str,
        default="record",
        help="Directory to store evaluation results",
    )
    parser.add_argument(
        "--error_jsonl_path",
        type=str,
        required=False,
        help="Path to the JSONL file to store error cases",
    )
    args = parser.parse_args()

    subfolder = os.path.basename(os.path.dirname(args.predictions_path))
    model_name = os.path.basename(args.predictions_path).split("_")[0]
    prompt_version = os.path.basename(args.predictions_path)  # If needed
    model_name = f"{model_name}_{prompt_version}"

    # Determine if error cases should be saved based on prompt type
    EA_flag = False
    if (
        "zero_shot" in args.predictions_path
        or "rag" in args.predictions_path
        or "intent" in args.predictions_path
    ):
        EA_flag = True

    # Load data
    gt_data = load_jsonl(args.ground_truth_path)
    print(f"Loaded {len(gt_data)} ground truth samples")

    predict_data = load_jsonl(args.predictions_path)
    print(f"Loaded {len(predict_data)} prediction samples")

    # Determine source and target from file naming
    if (
        "numpy_to_pandas" in args.predictions_path
        or "np_to_pd" in args.predictions_path
    ):
        source, target = "numpy", "pandas"
    else:
        source, target = "pandas", "numpy"

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
    )
