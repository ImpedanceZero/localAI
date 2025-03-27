import requests
import os
import time
import logging
import json
import sys
import argparse
import threading
import subprocess
import csv
import re
import random
import numpy as np
import selectors
from typing import Dict, Any, Optional, Literal, List, Tuple
from logging.handlers import RotatingFileHandler
from requests.exceptions import RequestException, Timeout, HTTPError
from requests.sessions import Session
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, ValidationError
from datetime import datetime
from statistics import mean, stdev, quantiles

# Define logger at the module level
logger = logging.getLogger("AnythingLLMTest")

# Custom exception for AnythingLLM-specific errors
class AnythingLLMError(Exception):
    """Base exception for AnythingLLM-related errors."""
    pass

class AnythingLLMConfigError(AnythingLLMError):
    """Raised when configuration is invalid."""
    pass

class AnythingLLMRequestError(AnythingLLMError):
    """Raised when a request to AnythingLLM fails."""
    pass

# Pydantic models for response validation
class Source(BaseModel):
    id: str
    url: str
    title: str
    docAuthor: Optional[str]
    description: Optional[str]
    docSource: Optional[str]
    chunkSource: Optional[str]
    published: Optional[str]
    wordCount: Optional[int]
    token_count_estimate: Optional[int]
    text: Optional[str]
    _distance: Optional[float]
    score: Optional[float]

class Metrics(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    outputTps: float
    duration: float

class AnythingLLMResponse(BaseModel):
    id: str
    type: str
    close: bool
    error: Optional[str]
    chatId: Optional[int]
    textResponse: Optional[str]
    sources: Optional[list[Source]]
    metrics: Optional[Metrics]

# Configuration class for managing global parameters
class Config:
    def __init__(self):
        self.base_url: str = os.getenv("ANYTHING_LLM_BASE_URL", "http://llm.blue.local")
        self.log_level: str = os.getenv("ANYTHING_LLM_LOG_LEVEL", "INFO")
        # Define log directory as a relative path
        self.log_dir: str = "log"
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except PermissionError as e:
            raise AnythingLLMConfigError(f"Failed to create log directory {self.log_dir}: {str(e)}")
        # Prepend log directory to all file paths
        self.log_file: str = os.path.join(self.log_dir, os.getenv("ANYTHING_LLM_LOG_FILE", "anything_llm_test.log"))
        self.log_max_bytes: int = int(os.getenv("ANYTHING_LLM_LOG_MAX_BYTES", 5 * 1024 * 1024))
        self.log_backup_count: int = int(os.getenv("ANYTHING_LLM_LOG_BACKUP_COUNT", 3))
        self.request_timeout: int = int(os.getenv("ANYTHING_LLM_REQUEST_TIMEOUT", 900))
        self.api_key: str = os.getenv("ANYTHING_LLM_API_KEY", "")
        # Number of users is fixed at 9 (for simulation purposes)
        self.num_users: int = 9
        self.nvidia_smi_log_file: str = os.path.join(self.log_dir, os.getenv("ANYTHING_LLM_NVIDIA_SMI_LOG", "nvidia_smi_metrics.csv"))
        # Define log files for both llama.cpp containers
        self.llama_cpp_containers: List[str] = ["poc-llama-cpp-codestral-1", "poc-llama-cpp-mistral-1"]
        self.llama_cpp_log_files: Dict[str, str] = {
            "poc-llama-cpp-codestral-1": os.path.join(self.log_dir, "llama_cpp_metrics_codestral.log"),
            "poc-llama-cpp-mistral-1": os.path.join(self.log_dir, "llama_cpp_metrics_mistral.log")
        }
        self.csv_output_file: str = os.path.join(self.log_dir, os.getenv("ANYTHING_LLM_CSV_OUTPUT", "performance_metrics.csv"))
        # Input CSV file for prompts and workspaces
        self.input_csv: str = "series3_input.csv"
        # Mean inter-request delay (seconds) for exponential distribution
        self.mean_inter_request_delay: float = 30.0  # Mean delay of 30 seconds

    def validate(self) -> None:
        if not self.api_key:
            raise AnythingLLMConfigError("API key not found. Please set the ANYTHING_LLM_API_KEY environment variable.")
        if self.num_users < 1:
            raise AnythingLLMConfigError("Number of users must be at least 1.")

# Headers (constructed once to avoid repeated dictionary creation)
HEADERS: Dict[str, str] = {
    "Authorization": f"Bearer {os.getenv('ANYTHING_LLM_API_KEY', '')}",
    "accept": "application/json",
    "Content-Type": "application/json"
}

def setup_logging(config: Config) -> None:
    """
    Configure logging with console and file handlers, including rotation.
    
    Args:
        config (Config): Configuration object with logging settings.
    """
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # Avoid adding handlers if they already exist (prevents duplicate logs)
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler with rotation
        try:
            file_handler = RotatingFileHandler(
                config.log_file,
                maxBytes=config.log_max_bytes,
                backupCount=config.log_backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {config.log_file}")
        except PermissionError as e:
            logger.error(f"Failed to set up file logging to {config.log_file}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error setting up file logging to {config.log_file}: {str(e)}")
            raise

def truncate_string(s: str, max_length: int = 1000) -> str:
    """
    Truncate a string to a maximum length to prevent log file bloat.
    
    Args:
        s (str): The string to truncate.
        max_length (int): Maximum length of the string.
    
    Returns:
        str: Truncated string with ellipsis if necessary.
    """
    if len(s) > max_length:
        return s[:max_length] + "... (truncated)"
    return s

def validate_csv(config: Config) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Validate the input CSV file and return valid and invalid rows.
    
    Args:
        config (Config): Configuration object with the input CSV path.
    
    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: (valid_rows, invalid_rows)
    """
    valid_rows = []
    invalid_rows = []
    try:
        # First, read the raw lines to check for empty files or missing headers
        with open(config.input_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"CSV file {config.input_csv} is empty")
            if len(lines) < 2:
                raise ValueError(f"CSV file {config.input_csv} has no data rows (only header or empty)")

        # Now use csv.DictReader to parse the CSV
        with open(config.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check if required columns exist
            required_columns = {'prompt', 'workspace'}
            if not reader.fieldnames:
                logger.error("CSV file has no header row")
                raise ValueError(f"CSV file {config.input_csv} has no header row. Expected header: 'prompt,workspace'")
            logger.debug(f"CSV header: {reader.fieldnames}")
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                logger.error(f"Found columns: {reader.fieldnames}")
                raise ValueError(f"CSV file {config.input_csv} is missing required columns: {missing}. Found columns: {reader.fieldnames}")
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                prompt = row.get('prompt', '').strip()
                workspace = row.get('workspace', '').strip()
                if not prompt or not workspace:
                    invalid_rows.append({
                        "row_num": row_num,
                        "prompt": prompt,
                        "workspace": workspace,
                        "error": "Empty prompt or workspace"
                    })
                else:
                    valid_rows.append({
                        "row_num": row_num,
                        "prompt": prompt,
                        "workspace": workspace
                    })
        return valid_rows, invalid_rows
    except FileNotFoundError:
        logger.error(f"Input CSV file {config.input_csv} not found")
        raise
    except Exception as e:
        logger.error(f"Failed to validate CSV {config.input_csv}: {str(e)}")
        raise

def load_prompts_from_csv(config: Config) -> List[Tuple[str, str]]:
    """
    Load pairs of prompts and workspace slugs from the input CSV file.
    
    Args:
        config (Config): Configuration object with the input CSV path.
    
    Returns:
        List[Tuple[str, str]]: List of (prompt, workspace_slug) pairs.
    """
    prompts = []
    valid_rows, invalid_rows = validate_csv(config)
    
    # Log invalid rows with row numbers
    if invalid_rows:
        logger.warning(f"Found {len(invalid_rows)} invalid rows in {config.input_csv}:")
        for invalid_row in invalid_rows:
            logger.warning(
                f"Row {invalid_row['row_num']}: prompt='{invalid_row['prompt']}', "
                f"workspace='{invalid_row['workspace']}', error='{invalid_row['error']}'"
            )
        # Optionally, raise an error if too many invalid rows are found
        invalid_ratio = len(invalid_rows) / (len(valid_rows) + len(invalid_rows))
        if invalid_ratio > 0.1:  # More than 10% invalid rows
            raise ValueError(
                f"Too many invalid rows in {config.input_csv} ({len(invalid_rows)} invalid, "
                f"{len(valid_rows)} valid). Please fix the CSV file and try again."
            )

    # Convert valid rows to list of (prompt, workspace) tuples
    for row in valid_rows:
        prompts.append((row['prompt'], row['workspace']))
    
    logger.info(f"Loaded {len(prompts)} valid prompt-workspace pairs from {config.input_csv}")
    return prompts

def capture_nvidia_smi_metrics(config: Config, stop_event: threading.Event) -> None:
    """
    Capture NVIDIA metrics using nvidia-smi in a background thread.
    
    Args:
        config (Config): Configuration object with NVIDIA log file settings.
        stop_event (threading.Event): Event to signal when to stop capturing metrics.
    """
    try:
        with open(config.nvidia_smi_log_file, 'w') as f:
            # Write CSV header
            f.write("timestamp,gpu_utilization,memory_utilization,memory_used,memory_total\n")
            f.flush()
            while not stop_event.is_set():
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total', '--format=csv,noheader,nounits'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if result.stderr:
                        logger.warning(f"nvidia-smi stderr: {result.stderr}")
                    if result.stdout:
                        # Parse the output (e.g., "1, 27, 19954, 24576")
                        metrics = result.stdout.strip().split(',')
                        if len(metrics) == 4:
                            timestamp = datetime.now().isoformat()
                            gpu_util, mem_util, mem_used, mem_total = [m.strip() for m in metrics]
                            f.write(f"{timestamp},{gpu_util},{mem_util},{mem_used},{mem_total}\n")
                            f.flush()
                    time.sleep(1)  # Capture metrics every second
                except subprocess.SubprocessError as e:
                    logger.warning(f"Failed to capture NVIDIA metrics: {str(e)}")
                    time.sleep(1)
        logger.info(f"NVIDIA metrics written to: {config.nvidia_smi_log_file}")
    except PermissionError as e:
        logger.error(f"Failed to write NVIDIA metrics to {config.nvidia_smi_log_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error capturing NVIDIA metrics: {str(e)}")

def capture_llama_cpp_logs(config: Config, stop_event: threading.Event) -> None:
    """
    Capture llama.cpp Docker logs from all specified containers in real-time, starting from the current point.
    
    Args:
        config (Config): Configuration object with llama.cpp log file settings.
        stop_event (threading.Event): Event to signal when to stop capturing logs.
    """
    processes = []
    sel = selectors.DefaultSelector()
    # Dictionary to store file objects for each container
    file_handles = {}

    # Open all log files before starting log capture
    for container in config.llama_cpp_containers:
        log_file = config.llama_cpp_log_files[container]
        try:
            # Check if the container is running
            result = subprocess.run(
                ['docker', 'ps', '-q', '-f', f'name={container}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if not result.stdout.strip():
                logger.error(f"Container {container} is not running. Please start the container and try again.")
                continue

            # Open the log file in line-buffered text mode (buffering=1)
            f = open(log_file, 'w', buffering=1)
            file_handles[container] = f

            # Capture only new logs (starting from now) using --tail 0
            process = subprocess.Popen(
                ['docker', 'logs', '--tail', '0', '-f', container],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((container, process))
            logger.info(f"Started capturing llama.cpp logs for {container} to {log_file}")

            # Register the stdout and stderr streams with the selector
            sel.register(process.stdout, selectors.EVENT_READ, (container, f, 'stdout'))
            sel.register(process.stderr, selectors.EVENT_READ, (container, f, 'stderr'))

        except subprocess.SubprocessError as e:
            logger.error(f"Failed to start capturing llama.cpp logs for {container}: {str(e)}")
            if container in file_handles:
                file_handles[container].close()
                del file_handles[container]

    # Use selectors to handle both stdout and stderr non-blocking
    while processes and not stop_event.is_set():
        for key, _ in sel.select(timeout=1):
            container, f, stream_type = key.data
            line = key.fileobj.readline()
            if line:
                # Add a timestamp to the log line
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]  # Same format as logger
                timestamped_line = f"[{timestamp}] {line}"
                # Write the timestamped line to the container-specific log file
                f.write(timestamped_line)
                f.flush()  # Ensure immediate write to disk

        # Check if all processes have terminated
        processes = [(container, process) for container, process in processes if process.poll() is None]

    # Terminate all log capture processes
    for container, process in processes:
        process.terminate()
        try:
            _, stderr = process.communicate(timeout=1)
            if stderr:
                logger.warning(f"docker logs stderr for {container}: {stderr}")
        except subprocess.TimeoutExpired:
            process.kill()
            logger.warning(f"Timed out waiting for docker logs process for {container} to terminate; killed.")
        logger.info(f"Stopped capturing llama.cpp logs for {container}")

    # Close all file handles
    for container, f in file_handles.items():
        try:
            f.close()
            logger.debug(f"Closed log file for {container}")
            # Debug: Print the contents of the log file
            with open(config.llama_cpp_log_files[container], 'r') as f_read:
                log_contents = f_read.read()
                if log_contents:
                    logger.debug(f"Contents of {config.llama_cpp_log_files[container]}:\n{log_contents}")
                else:
                    logger.warning(f"Log file {config.llama_cpp_log_files[container]} is empty. The container may not have produced logs during the test.")
        except Exception as e:
            logger.error(f"Failed to close or read log file for {container}: {str(e)}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    retry=retry_if_exception_type((RequestException, Timeout, HTTPError))
)
def send_chat_request(
    session: Session,
    config: Config,
    prompt: str,
    workspace_slug: str,
    user_id: int,
    iteration: int
) -> Dict[str, Any]:
    """
    Send a chat request to the AnythingLLM API and return the result.
    
    Args:
        session (Session): Requests session for connection reuse.
        config (Config): Configuration object with API settings.
        prompt (str): The prompt to send.
        workspace_slug (str): The workspace slug to send the prompt to.
        user_id (int): Simulated user ID.
        iteration (int): Iteration number for this user.
    
    Returns:
        Dict[str, Any]: Result containing status, latency, and response data.
    """
    start_time = time.time()
    endpoint = f"{config.base_url}/api/v1/workspace/{workspace_slug}/chat"
    payload = {
        "message": prompt,
        "mode": "chat"
    }
    # Log the client request with a timestamp, request ID, user ID, prompt, and workspace
    logger.info(
        f"CLIENT_REQUEST - RequestID: {iteration}, UserID: {user_id}, "
        f"Prompt: '{truncate_string(prompt)}', Workspace: '{workspace_slug}'"
    )
    try:
        response = session.post(
            endpoint,
            headers=HEADERS,
            json=payload,
            timeout=config.request_timeout
        )
        response.raise_for_status()
        data = response.json()
        # Validate response using Pydantic
        validated_response = AnythingLLMResponse(**data)
        latency_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        result = {
            "user_id": user_id,
            "iteration": iteration,
            "status": "Success",
            "latency_ms": latency_ms,
            "response": validated_response.dict(),
            "metrics": validated_response.metrics.dict() if validated_response.metrics else {}
        }
        logger.info(f"User {user_id} (Iteration {iteration}): Request successful")
        return result
    except (ValidationError, ValueError) as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"User {user_id} (Iteration {iteration}): Response validation failed: {str(e)}")
        return {
            "user_id": user_id,
            "iteration": iteration,
            "status": "Failed",
            "latency_ms": latency_ms,
            "error": f"Response validation failed: {str(e)}"
        }
    except (RequestException, Timeout, HTTPError) as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"User {user_id} (Iteration {iteration}): Request failed: {str(e)}")
        return {
            "user_id": user_id,
            "iteration": iteration,
            "status": "Failed",
            "latency_ms": latency_ms,
            "error": str(e)
        }

def simulate_users(
    config: Config,
    session: Session,
    prompts: List[Tuple[str, str]],
    results: List[Dict[str, Any]]
) -> None:
    """
    Simulate 9 users submitting requests with realistic timing using an exponential distribution.
    
    Args:
        config (Config): Configuration object with API settings.
        session (Session): Requests session for connection reuse.
        prompts (List[Tuple[str, str]]): List of (prompt, workspace_slug) pairs.
        results (List[Dict[str, Any]]): Shared list to store results.
    """
    total_prompts = len(prompts)
    for i in range(total_prompts):
        # Assign the request to a random user (1 to 9)
        user_id = random.randint(1, config.num_users)
        prompt, workspace_slug = prompts[i]
        iteration = i + 1  # Global iteration number for logging

        logger.debug(f"User {user_id} (Iteration {iteration}): Submitting prompt '{truncate_string(prompt)}' to workspace '{workspace_slug}'")
        result = send_chat_request(session, config, prompt, workspace_slug, user_id, iteration)
        results.append(result)

        # Generate an inter-request delay using an exponential distribution
        # Mean delay is config.mean_inter_request_delay (e.g., 30 seconds)
        # Enforce a minimum delay of 10 seconds and a maximum of 5 minutes (300 seconds)
        delay = max(10, min(300, np.random.exponential(config.mean_inter_request_delay)))
        logger.debug(f"Waiting {delay:.2f} seconds before the next request")
        time.sleep(delay)

def parse_nvidia_smi_metrics(config: Config) -> List[Dict[str, Any]]:
    """
    Parse NVIDIA metrics from the log file.
    
    Args:
        config (Config): Configuration object with NVIDIA log file settings.
    
    Returns:
        List[Dict[str, Any]]: List of parsed NVIDIA metrics.
    """
    metrics = []
    try:
        with open(config.nvidia_smi_log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    metrics.append({
                        "timestamp": row["timestamp"],
                        "gpu_utilization": float(row["gpu_utilization"]),
                        "memory_utilization": float(row["memory_utilization"]),
                        "memory_used": float(row["memory_used"]),
                        "memory_total": float(row["memory_total"])
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse NVIDIA metrics row: {row}, error: {str(e)}")
        return metrics
    except FileNotFoundError:
        logger.warning(f"NVIDIA metrics file {config.nvidia_smi_log_file} not found")
        return []
    except Exception as e:
        logger.error(f"Failed to parse NVIDIA metrics: {str(e)}")
        return []

def parse_llama_cpp_logs(config: Config) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse llama.cpp logs to extract performance metrics.
    
    Args:
        config (Config): Configuration object with llama.cpp log file settings.
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping container names to lists of metrics.
    """
    metrics = {container: [] for container in config.llama_cpp_containers}
    # Updated pattern to match timestamped log lines
    prompt_eval_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] prompt eval time\s+=\s+(\d+\.\d+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+(\d+\.\d+)\s+ms per token,\s+(\d+\.\d+)\s+tokens per second\)"
    )
    eval_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] eval time\s+=\s+(\d+\.\d+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+(\d+\.\d+)\s+ms per token,\s+(\d+\.\d+)\s+tokens per second\)"
    )
    total_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] total time\s+=\s+(\d+\.\d+)\s+ms\s+/\s+(\d+)\s+tokens"
    )

    for container in config.llama_cpp_containers:
        log_file = config.llama_cpp_log_files[container]
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                # Parse prompt eval metrics
                for match in prompt_eval_pattern.finditer(log_content):
                    timestamp, time_ms, tokens, ms_per_token, tps = match.groups()
                    metrics[container].append({
                        "timestamp": timestamp,
                        "type": "prompt_eval",
                        "time_ms": float(time_ms),
                        "tokens": int(tokens),
                        "ms_per_token": float(ms_per_token),
                        "tps": float(tps)
                    })
                # Parse eval metrics
                for match in eval_pattern.finditer(log_content):
                    timestamp, time_ms, tokens, ms_per_token, tps = match.groups()
                    metrics[container].append({
                        "timestamp": timestamp,
                        "type": "eval",
                        "time_ms": float(time_ms),
                        "tokens": int(tokens),
                        "ms_per_token": float(ms_per_token),
                        "tps": float(tps)
                    })
                # Parse total metrics
                for match in total_pattern.finditer(log_content):
                    timestamp, time_ms, tokens = match.groups()
                    metrics[container].append({
                        "timestamp": timestamp,
                        "type": "total",
                        "time_ms": float(time_ms),
                        "tokens": int(tokens)
                    })
        except FileNotFoundError:
            logger.warning(f"llama.cpp log file {log_file} not found")
        except Exception as e:
            logger.error(f"Failed to parse llama.cpp logs for {container}: {str(e)}")
    return metrics

def save_metrics_to_csv(
    config: Config,
    results: List[Dict[str, Any]],
    nvidia_metrics: List[Dict[str, Any]],
    llama_metrics: Dict[str, List[Dict[str, Any]]]
) -> None:
    """
    Save all metrics to a CSV file.
    
    Args:
        config (Config): Configuration object with CSV output file settings.
        results (List[Dict[str, Any]]): List of AnythingLLM request results.
        nvidia_metrics (List[Dict[str, Any]]): List of NVIDIA metrics.
        llama_metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of llama.cpp metrics.
    """
    try:
        with open(config.csv_output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write AnythingLLM Metrics
            writer.writerow(["AnythingLLM Metrics"])
            writer.writerow([
                "User ID", "Iteration", "Status", "Latency (ms)", "Prompt Tokens",
                "Completion Tokens", "Total Tokens", "Output TPS (tokens/s)", "Duration (s)"
            ])
            for result in results:
                metrics = result.get("metrics", {})
                writer.writerow([
                    result["user_id"],
                    result["iteration"],
                    result["status"],
                    f"{result['latency_ms']:.2f}",
                    metrics.get("prompt_tokens", 0),
                    metrics.get("completion_tokens", 0),
                    metrics.get("total_tokens", 0),
                    metrics.get("outputTps", 0),
                    metrics.get("duration", 0)
                ])
            writer.writerow([])

            # Write NVIDIA Metrics
            writer.writerow(["NVIDIA Metrics"])
            writer.writerow(["Timestamp", "GPU Utilization (%)", "Memory Utilization (%)", "Memory Used (MiB)", "Memory Total (MiB)"])
            for metric in nvidia_metrics:
                writer.writerow([
                    metric["timestamp"],
                    metric["gpu_utilization"],
                    metric["memory_utilization"],
                    metric["memory_used"],
                    metric["memory_total"]
                ])
            writer.writerow([])

            # Write llama.cpp Metrics
            for container, container_metrics in llama_metrics.items():
                writer.writerow([f"llama CPP Metrics ({container})"])
                writer.writerow(["Timestamp", "Type", "Time (ms)", "Tokens", "TPS (tokens/s)"])
                for metric in container_metrics:
                    writer.writerow([
                        metric["timestamp"],
                        metric["type"],
                        f"{metric['time_ms']:.2f}" if "time_ms" in metric else "",
                        metric["tokens"],
                        f"{metric['tps']:.2f}" if "tps" in metric else ""
                    ])
                writer.writerow([])
        logger.info(f"Metrics saved to CSV: {config.csv_output_file}")
    except PermissionError as e:
        logger.error(f"Failed to write metrics to CSV {config.csv_output_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error saving metrics to CSV: {str(e)}")

def main() -> None:
    # Parse command-line arguments (log level only, since num_users is fixed)
    parser = argparse.ArgumentParser(description="AnythingLLM Performance Test Script")
    parser.add_argument("--log-level", type=str, default=None, help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    if args.log_level:
        config.log_level = args.log_level
    config.validate()

    # Setup logging
    setup_logging(config)

    # Load prompts from CSV
    prompts = load_prompts_from_csv(config)
    if not prompts:
        logger.error("No valid prompts loaded from CSV. Exiting.")
        sys.exit(1)

    logger.info(f"Starting performance test with {config.num_users} simulated users, {len(prompts)} total prompts")

    # Start capturing NVIDIA metrics in a background thread
    stop_event = threading.Event()

    nvidia_thread = threading.Thread(target=capture_nvidia_smi_metrics, args=(config, stop_event))
    nvidia_thread.start()

    # Simulate users submitting requests
    results = []
    with Session() as session:
        # Start capturing llama.cpp logs just before sending the first request
        llama_cpp_thread = threading.Thread(target=capture_llama_cpp_logs, args=(config, stop_event))
        llama_cpp_thread.start()

        try:
            simulate_users(config, session, prompts, results)
        finally:
            # Stop capturing NVIDIA metrics and llama.cpp logs
            stop_event.set()
            nvidia_thread.join()
            llama_cpp_thread.join()

    # Analyze results
    successful_results = [r for r in results if r['status'] == "Success"]
    failed_results = [r for r in results if r['status'] != "Success"]

    logger.info("\nPerformance Test Summary:")
    logger.info(f"Total Requests: {len(results)}")
    logger.info(f"Successful Requests: {len(successful_results)}")
    logger.info(f"Failed Requests: {len(failed_results)}")

    # Parse and summarize AnythingLLM metrics
    if successful_results:
        latencies = [r['latency_ms'] for r in successful_results]
        prompt_tokens = [r['metrics']['prompt_tokens'] for r in successful_results if 'metrics' in r and 'prompt_tokens' in r['metrics']]
        completion_tokens = [r['metrics']['completion_tokens'] for r in successful_results if 'metrics' in r and 'completion_tokens' in r['metrics']]
        total_tokens = [r['metrics']['total_tokens'] for r in successful_results if 'metrics' in r and 'total_tokens' in r['metrics']]
        output_tps = [r['metrics']['outputTps'] for r in successful_results if 'metrics' in r and 'outputTps' in r['metrics']]
        durations = [r['metrics']['duration'] for r in successful_results if 'metrics' in r and 'duration' in r['metrics']]

        # Calculate percentiles for latency
        latency_percentiles = quantiles(latencies, n=10) if len(latencies) >= 10 else latencies

        logger.info("\nPerformance Metrics:")
        logger.info(f"Response Latency (Total): {mean(latencies):.2f} ms (Std Dev: {stdev(latencies) if len(latencies) > 1 else 0:.2f} ms)")
        logger.info(f"Response Latency Percentiles (10th, 50th, 90th): {latency_percentiles[0]:.2f} ms, {latency_percentiles[4]:.2f} ms, {latency_percentiles[8]:.2f} ms")

    # Parse and summarize NVIDIA metrics
    nvidia_metrics = parse_nvidia_smi_metrics(config)
    if nvidia_metrics:
        gpu_utilizations = [m['gpu_utilization'] for m in nvidia_metrics]
        memory_utilizations = [m['memory_utilization'] for m in nvidia_metrics]
        logger.info(f"GPU Utilization: {mean(gpu_utilizations):.2f}% (Std Dev: {stdev(gpu_utilizations) if len(gpu_utilizations) > 1 else 0:.2f}%)")
        logger.info(f"Memory Utilization: {mean(memory_utilizations):.2f}% (Std Dev: {stdev(memory_utilizations) if len(memory_utilizations) > 1 else 0:.2f}%)")

    # Parse and summarize llama.cpp metrics from all containers
    llama_metrics = parse_llama_cpp_logs(config)
    for container, metrics in llama_metrics.items():
        prompt_eval_metrics = [m for m in metrics if m['type'] == "prompt_eval"]
        eval_metrics = [m for m in metrics if m['type'] == "eval"]
        total_metrics = [m for m in metrics if m['type'] == "total"]

        logger.info(f"\nllama.cpp Metrics for {container}:")
        if prompt_eval_metrics:
            prompt_eval_tps = [m['tps'] for m in prompt_eval_metrics]
            prompt_eval_times = [m['time_ms'] for m in prompt_eval_metrics]
            logger.info(f"Prompt Eval TPS: {mean(prompt_eval_tps):.2f} tokens/s (Std Dev: {stdev(prompt_eval_tps) if len(prompt_eval_tps) > 1 else 0:.2f})")
            logger.info(f"Response Latency (TTFT, approximated): {mean(prompt_eval_times):.2f} ms (Std Dev: {stdev(prompt_eval_times) if len(prompt_eval_times) > 1 else 0:.2f})")
        else:
            logger.warning(f"No prompt eval metrics found in {container} logs.")

        if eval_metrics:
            eval_tps = [m['tps'] for m in eval_metrics]
            logger.info(f"Generation TPS: {mean(eval_tps):.2f} tokens/s (Std Dev: {stdev(eval_tps) if len(eval_tps) > 1 else 0:.2f})")
        else:
            logger.warning(f"No eval metrics found in {container} logs.")

    # Save metrics to CSV
    save_metrics_to_csv(config, results, nvidia_metrics, llama_metrics)

if __name__ == "__main__":
    main()