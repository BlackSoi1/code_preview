import sys
import os
import json
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import datetime
import warnings

from execution import check_correctness

warnings.filterwarnings("ignore")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, one per line.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Append a list of dictionaries to a JSONL file, one JSON object per line.

    Args:
        data (List[Dict[str, Any]]): Data to append.
        file_path (str): Path to the output JSONL file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def extract_errors(
    results: List[Dict[str, Any]],
    gt_data: List[Dict[str, Any]],
    predict_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract error cases from the evaluation results.

    Args:
        results (List[Dict[str, Any]]): Results of evaluation for each prediction.
        gt_data (List[Dict[str, Any]]): Ground truth data.
        predict_data (List[Dict[str, Any]]): Predicted data.

    Returns:
        List[Dict[str, Any]]: A list of error case entries.
    """
    errors = []
    for i, res in enumerate(results):
        if not res["passed"]:
            error_entry = {
                "ground_truth": gt_data[i],
                "prediction": predict_data[i],
                "error_result": res["result"],
            }
            errors.append(error_entry)
    return errors


def calculate_accuracy(
    results: List[Dict[str, Any]],
    source: str,
    target: str,
    model_name: str,
    subfolder: str,
    record_folder: str,
) -> float:
    """
    Calculate and record the accuracy and various error statistics of the evaluation.

    Args:
        results (List[Dict[str, Any]]): Evaluation results.
        source (str): Source framework/library name.
        target (str): Target framework/library name.
        model_name (str): Name of the model used for generation.
        subfolder (str): Subfolder name (e.g., version or prompt type).
        record_folder (str): Directory to store the record files.

    Returns:
        float: The calculated accuracy percentage.
    """
    num_queries = len(results)
    correct_count = sum(1 for res in results if res["passed"])
    execution_errors = sum(1 for res in results if res["result"].startswith("failed:"))
    assert_errors = sum(
        1 for res in results if res["result"].startswith("Assertion failed:")
    )
    timeout_errors = sum(1 for res in results if res["result"] == "timed out")
    accuracy = (correct_count / num_queries) * 100

    # Construct statistics message
    stats_message = f"Model: {model_name}\n"
    stats_message += f"\n{source} To {target} result statistics:\n"
    stats_message += f"Number of Execution Errors: {execution_errors}\n"
    stats_message += f"Number of Timeouts: {timeout_errors}\n"
    stats_message += f"Number of Assertion Errors: {assert_errors}\n"
    stats_message += f"Total Errors: {num_queries - correct_count}\n"
    stats_message += f"{source} To {target} Accuracy: {accuracy:.2f}%\n"
    stats_message += f"Timestamp: {datetime.datetime.now()}\n"

    print(stats_message)

    # Determine record file name based on source
    if "pytorch" in source:
        record_file_name = f"py_to_tf_{subfolder}.txt"
    elif "numpy" in source:
        record_file_name = f"np_to_pd_{subfolder}.txt"
    else:
        record_file_name = f"tf_to_py_{subfolder}.txt"

    record_file_path = os.path.join(record_folder, record_file_name)
    os.makedirs(record_folder, exist_ok=True)

    # Create record file if it doesn't exist, then append stats
    if not os.path.exists(record_file_path):
        with open(record_file_path, "w", encoding="utf-8") as f:
            f.write("Record file created.\n")
            f.write("-" * 50 + "\n")

    try:
        with open(record_file_path, "a", encoding="utf-8") as f:
            f.write(stats_message)
            f.write("-" * 50 + "\n")
    except IOError as e:
        print(f"Error writing to file {record_file_path}: {e}")

    return accuracy


def format_test_program(
    library: str,
    target_test: Dict[str, Any],
    start_code: str,
    sol_code: str,
    target: str,
) -> str:
    """
    Construct the test program by combining setup code, solution code, and test cases.

    Args:
        library (str): Import or setup lines for the library.
        target_test (Dict[str, Any]): Dictionary containing setup code and test cases.
        start_code (str): Starting code snippet.
        sol_code (str): Solution code snippet.
        target (str): Target framework name.

    Returns:
        str: The complete test program as a string.
    """
    program = ""
    if target == "tensorflow":
        # Suppress TensorFlow warnings and logs
        program += (
            "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
            "from absl import logging\nlogging.set_verbosity(logging.ERROR)\n"
        )

    program += f"{library}\n{start_code}\n{sol_code}\n{target_test['setup_code']}\n"
    for test_case in target_test["test_cases"]:
        program += f"{test_case}\n"
    return program


def evaluate_single_prediction(
    target: str,
    library: str,
    target_test: Dict[str, Any],
    start_code: str,
    pred_code: str,
    timeout: int,
) -> Dict[str, Any]:
    """
    Evaluate a single prediction by constructing the program and running correctness checks.

    Args:
        target (str): Target framework name.
        library (str): Import/setup code for the library.
        target_test (Dict[str, Any]): Contains setup code and test cases.
        start_code (str): Starting code snippet for the given scenario.
        pred_code (str): The predicted solution code.
        timeout (int): Maximum time allowed for execution.

    Returns:
        Dict[str, Any]: A dictionary containing 'passed' status and 'result' message.
    """
    program = format_test_program(library, target_test, start_code, pred_code, target)
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
    Evaluate all predictions using parallel workers. Collect error cases if requested and calculate accuracy.

    Args:
        gt_data (List[Dict[str, Any]]): Ground truth data.
        predict_data (List[Dict[str, Any]]): Prediction data.
        source (str): Source framework/library name.
        target (str): Target framework/library name.
        timeout (int): Execution timeout in seconds.
        max_workers (int): Number of parallel threads.
        model_name (str): Name of the model.
        subfolder (str): Subfolder/version identifier.
        record_folder (str): Directory to store result records.
        error_jsonl_path (str): Path to store error cases.
        EA_flag (bool): Whether to write error cases to a JSONL file.

    Returns:
        float: The accuracy of the predictions.
    """
    results = []
    error_cases = []

    # Prepare test evaluation tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_prediction,
                target,
                gt[f"{target}_library"],
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

        # Process futures and gather results
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
                "prediction": predict_data[idx],
                "passed": res["passed"],
                "error_result": res["result"],
            }
            error_cases.append(error_case)

    # Optionally save error cases
    if error_cases and EA_flag:
        save_jsonl(error_cases, error_jsonl_path)

    # Calculate and record accuracy
    accuracy = calculate_accuracy(
        results, source, target, model_name, subfolder, record_folder
    )
    return accuracy


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
        help="Timeout for each prediction evaluation in seconds",
    )
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--record_folder",
        type=str,
        default="record",
        help="Folder to store evaluation results",
    )
    parser.add_argument(
        "--error_jsonl_path",
        type=str,
        required=False,
        help="Path to the JSONL file to store error cases",
    )
    args = parser.parse_args()

    # Determine subfolder and model name from predictions_path
    subfolder = os.path.basename(os.path.dirname(args.predictions_path))
    model_name = os.path.basename(args.predictions_path).split("_")[0]
    prompt_version = os.path.basename(args.predictions_path)
    model_name = f"{model_name}_{prompt_version}"

    # Decide whether to save error cases (EA_flag)
    EA_flag = False
    if (
        "zero_shot" in args.predictions_path
        and "zero_shot_cot" not in args.predictions_path
    ):
        EA_flag = True

    # Load ground truth and predictions
    gt_data = load_jsonl(args.ground_truth_path)
    print(f"Loaded {len(gt_data)} ground truth samples")
    predict_data = load_jsonl(args.predictions_path)
    print(f"Loaded {len(predict_data)} prediction samples")

    # Determine source and target based on filename
    if (
        "pytorch_to_tensorflow" in args.predictions_path
        or "py_to_tf" in args.predictions_path
    ):
        source, target = "pytorch", "tensorflow"
    else:
        source, target = "tensorflow", "pytorch"

    # Evaluate predictions and print accuracy
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
