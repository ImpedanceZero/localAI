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
import math
from typing import Dict, Any, Optional, Literal, List
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
        self.test_prompt: str = os.getenv("ANYTHING_LLM_TEST_PROMPT", "How many calories are in a Quiznos Classic Italian sub?")
        self.test_workspace: str = os.getenv("ANYTHING_LLM_TEST_WORKSPACE", "docexplorer-document-rag-chat")
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
        self.request_timeout: int = int(os.getenv("ANYTHING_LLM_REQUEST_TIMEOUT", 30))
        self.api_key: str = os.getenv("ANYTHING_LLM_API_KEY", "")
        self.chat_endpoint: str = f"{self.base_url}/api/v1/workspace/{self.test_workspace}/chat"
        self.num_users: int = int(os.getenv("ANYTHING_LLM_NUM_USERS", 3))
        self.num_iterations: int = int(os.getenv("ANYTHING_LLM_NUM_ITERATIONS", 50))
        self.nvidia_smi_log_file: str = os.path.join(self.log_dir, os.getenv("ANYTHING_LLM_NVIDIA_SMI_LOG", "nvidia_smi_metrics.csv"))
        # Define log files for both llama.cpp containers
        self.llama_cpp_containers: List[str] = ["poc-llama-cpp-codestral-1", "poc-llama-cpp-mistral-1"]
        self.llama_cpp_log_files: Dict[str, str] = {
            "poc-llama-cpp-codestral-1": os.path.join(self.log_dir, "llama_cpp_metrics_codestral.log"),
            "poc-llama-cpp-mistral-1": os.path.join(self.log_dir, "llama_cpp_metrics_mistral.log")
        }
        self.csv_output_file: str = os.path.join(self.log_dir, os.getenv("ANYTHING_LLM_CSV_OUTPUT", "performance_metrics.csv"))
        self.markdown_output_file: str = os.path.join(self.log_dir, "results.md")

    def validate(self) -> None:
        if not self.api_key:
            raise AnythingLLMConfigError("API key not found. Please set the ANYTHING_LLM_API_KEY environment variable.")
        if self.num_users < 1:
            raise AnythingLLMConfigError("Number of users must be at least 1.")
        if self.num_iterations < 1:
            raise AnythingLLMConfigError("Number of iterations must be at least 1.")

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

def capture_nvidia_smi_metrics(config: Config, stop_event: threading.Event) -> None:
    """
    Capture NVIDIA metrics using nvidia-smi in a background thread.
    
    Args:
        config (Config): Configuration object with NVIDIA log file settings.
        stop_event (threading.Event): Event to signal when to stop capturing metrics.
    """
    try:
        with open(config.nvidia_smi_log_file, 'w') as f:
            process = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total', '--format=csv', '-l', '1'],
                stdout=f,
                stderr=subprocess.PIPE,
                text=True
            )
            while not stop_event.is_set():
                time.sleep(1)
            process.terminate()
            _, stderr = process.communicate()
            if stderr:
                logger.warning(f"nvidia-smi stderr: {stderr}")
        logger.info(f"NVIDIA metrics written to: {config.nvidia_smi_log_file}")
    except PermissionError as e:
        logger.error(f"Failed to write NVIDIA metrics to {config.nvidia_smi_log_file}: {str(e)}")
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to capture NVIDIA metrics: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error capturing NVIDIA metrics: {str(e)}")

def capture_llama_cpp_logs(config: Config, stop_event: threading.Event) -> None:
    """
    Capture llama.cpp Docker logs from all specified containers in real-time.
    
    Args:
        config (Config): Configuration object with llama.cpp log file settings.
        stop_event (threading.Event): Event to signal when to stop capturing logs.
    """
    processes = []
    for container in config.llama_cpp_containers:
        log_file = config.llama_cpp_log_files[container]
        try:
            # Open the log file in append mode to capture both stdout and stderr
            with open(log_file, 'w') as f:
                # Use docker logs -f -t to follow logs in real-time with timestamps
                process = subprocess.Popen(
                    ['docker', 'logs', '-f', '-t', container],
                    stdout=f,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    text=True
                )
                processes.append((container, process))
                logger.info(f"Started capturing llama.cpp logs for {container} to {log_file}")
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to start capturing llama.cpp logs for {container}: {str(e)}")
            continue
    
    # Wait until the stop event is set
    while not stop_event.is_set():
        time.sleep(1)
    
    # Terminate all log capture processes
    for container, process in processes:
        process.terminate()
        process.wait()  # Ensure the process has terminated
        logger.info(f"Stopped capturing llama.cpp logs for {container}")
        
        # Debug: Print the contents of the log file
        try:
            with open(log_file, 'r') as f:
                log_contents = f.read()
                logger.debug(f"Contents of {log_file}:\n{log_contents}")
        except Exception as e:
            logger.error(f"Failed to read log file {log_file}: {str(e)}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Timeout, HTTPError)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying request (attempt {retry_state.attempt_number}/3) after {retry_state.idle_for:.2f}s..."
    )
)
def submit_query(prompt: str, session: Session, config: Config, user_id: int, iteration: int) -> Dict[str, Any]:
    """
    Submit a single query to AnythingLLM in the specified workspace and return the response.
    
    Args:
        prompt (str): The query to submit.
        session (Session): HTTP session for connection reuse.
        config (Config): Configuration object with API settings.
        user_id (int): User ID for logging purposes.
        iteration (int): Iteration number for logging purposes.
    
    Returns:
        Dict[str, Any]: The response from the server, including status and data.
    
    Raises:
        AnythingLLMRequestError: If the request fails after retries.
    """
    # Payload for the chat request
    payload = {
        "message": prompt,
        "mode": "chat"
    }
    
    try:
        # Record start time for latency measurement
        start_time = time.time()
        
        # Send POST request to the chat endpoint
        logger.debug(f"User {user_id} (Iteration {iteration}): Sending request to {config.chat_endpoint}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"User {user_id} (Iteration {iteration}): Request payload: {json.dumps(payload, indent=2)}")
        response = session.post(
            config.chat_endpoint,
            json=payload,
            headers=HEADERS,
            timeout=config.request_timeout
        )
        
        # Calculate latency in milliseconds
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Raise an exception for non-200 status codes
        response.raise_for_status()
        
        # Log response details
        logger.debug(f"User {user_id} (Iteration {iteration}): Response status code: {response.status_code}")
        logger.debug(f"User {user_id} (Iteration {iteration}): Response headers: {response.headers}")
        logger.debug(f"User {user_id} (Iteration {iteration}): Raw Response: {truncate_string(response.text)}")
        
        # Check Content-Type to ensure it's JSON
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            logger.warning(f"User {user_id} (Iteration {iteration}): Response Content-Type is not JSON: {content_type}")
            raise AnythingLLMRequestError(f"Expected JSON response, got Content-Type: {content_type}")
        
        # Parse response as JSON
        response_data = response.json()
        
        # Validate response structure with Pydantic
        validated_response = AnythingLLMResponse(**response_data)
        
        logger.info(f"User {user_id} (Iteration {iteration}): Request successful")
        return {
            "status": "Success",
            "latency_ms": latency,
            "response": validated_response.dict()
        }
    
    except Timeout as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"User {user_id} (Iteration {iteration}): Request timed out after {config.request_timeout} seconds: {str(e)}")
        raise AnythingLLMRequestError(f"Request timed out after {config.request_timeout} seconds: {str(e)}")
    except HTTPError as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"User {user_id} (Iteration {iteration}): HTTP error occurred: {str(e)}")
        raise AnythingLLMRequestError(f"HTTP error: {str(e)}")
    except RequestException as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"User {user_id} (Iteration {iteration}): Request failed with exception: {str(e)}")
        raise AnythingLLMRequestError(f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"User {user_id} (Iteration {iteration}): Failed to parse JSON response: {str(e)}")
        raise AnythingLLMRequestError(f"Failed to parse JSON response: {str(e)}")
    except ValidationError as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"User {user_id} (Iteration {iteration}): Response validation failed: {str(e)}")
        raise AnythingLLMRequestError(f"Response validation failed: {str(e)}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Performance testing script for AnythingLLM API.")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides ANYTHING_LLM_LOG_LEVEL environment variable)"
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=None,
        help="Number of concurrent users (overrides ANYTHING_LLM_NUM_USERS environment variable)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Number of iterations per user (overrides ANYTHING_LLM_NUM_ITERATIONS environment variable)"
    )
    return parser.parse_args()

def parse_nvidia_smi_metrics(config: Config) -> List[Dict[str, Any]]:
    """
    Parse NVIDIA metrics from the nvidia-smi log file.
    
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
                # Strip whitespace from column names to handle variations
                row = {k.strip(): v for k, v in row.items()}
                metrics.append({
                    "timestamp": datetime.now().isoformat(),
                    "gpu_utilization": int(row['utilization.gpu [%]'].replace(' %', '')),
                    "memory_utilization": int(row['utilization.memory [%]'].replace(' %', '')),
                    "memory_used_mib": int(row['memory.used [MiB]'].replace(' MiB', '')),
                    "memory_total_mib": int(row['memory.total [MiB]'].replace(' MiB', ''))
                })
    except FileNotFoundError:
        logger.warning(f"NVIDIA metrics file not found: {config.nvidia_smi_log_file}")
    except KeyError as e:
        logger.error(f"Failed to parse NVIDIA metrics: Missing or unexpected column {str(e)}")
        logger.debug(f"Available columns: {list(row.keys())}")
    except Exception as e:
        logger.error(f"Failed to parse NVIDIA metrics: {str(e)}")
    return metrics

def parse_llama_cpp_logs(config: Config) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse llama.cpp metrics from all Docker log files.
    
    Args:
        config (Config): Configuration object with llama.cpp log file settings.
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping container names to their parsed metrics.
    """
    all_metrics = {}
    for container, log_file in config.llama_cpp_log_files.items():
        metrics = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Updated regex to match docker logs -t timestamp format and variable spacing
                    prompt_eval_match = re.match(
                        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,9}Z)\s+prompt eval time =\s+([\d.]+) ms / \s*(\d+) tokens \(.*?, \s*([\d.]+) tokens per second\)', line
                    )
                    eval_match = re.match(
                        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,9}Z)\s+eval time =\s+([\d.]+) ms / \s*(\d+) tokens \(.*?, \s*([\d.]+) tokens per second\)', line
                    )
                    total_match = re.match(
                        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,9}Z)\s+total time =\s+([\d.]+) ms / \s*(\d+) tokens', line
                    )
                    if prompt_eval_match:
                        metrics.append({
                            "timestamp": prompt_eval_match.group(1),
                            "type": "prompt_eval",
                            "time_ms": float(prompt_eval_match.group(2)),
                            "tokens": int(prompt_eval_match.group(3)),
                            "tps": float(prompt_eval_match.group(4))
                        })
                    elif eval_match:
                        metrics.append({
                            "timestamp": eval_match.group(1),
                            "type": "eval",
                            "time_ms": float(eval_match.group(2)),
                            "tokens": int(eval_match.group(3)),
                            "tps": float(eval_match.group(4))
                        })
                    elif total_match:
                        metrics.append({
                            "timestamp": total_match.group(1),
                            "type": "total",
                            "time_ms": float(total_match.group(2)),
                            "tokens": int(total_match.group(3))
                        })
        except FileNotFoundError:
            logger.warning(f"llama.cpp log file not found: {log_file}")
        except Exception as e:
            logger.error(f"Failed to parse llama.cpp logs for {container}: {str(e)}")
        all_metrics[container] = metrics
    return all_metrics

def run_user(config: Config, session: Session, user_id: int, results: List[Dict[str, Any]]) -> None:
    """
    Simulate a single user submitting multiple requests.
    
    Args:
        config (Config): Configuration object with test settings.
        session (Session): HTTP session for connection reuse.
        user_id (int): User ID for logging purposes.
        results (List[Dict[str, Any]]): List to store results from all users.
    """
    for iteration in range(config.num_iterations):
        try:
            result = submit_query(config.test_prompt, session, config, user_id, iteration + 1)
            results.append({
                "user_id": user_id,
                "iteration": iteration + 1,
                "status": result['status'],
                "latency_ms": result['latency_ms'],
                "metrics": result['response'].get('metrics', {})
            })
        except AnythingLLMRequestError as e:
            logger.error(f"User {user_id} (Iteration {iteration + 1}): Failed to submit query: {str(e)}")
            results.append({
                "user_id": user_id,
                "iteration": iteration + 1,
                "status": "Error",
                "latency_ms": None,
                "metrics": {}
            })

def save_metrics_to_csv(config: Config, results: List[Dict[str, Any]], nvidia_metrics: List[Dict[str, Any]], llama_metrics: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Save performance metrics to a CSV file.
    
    Args:
        config (Config): Configuration object with CSV output file settings.
        results (List[Dict[str, Any]]): List of AnythingLLM request results.
        nvidia_metrics (List[Dict[str, Any]]): List of NVIDIA metrics.
        llama_metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of llama.cpp metrics for each container.
    """
    try:
        with open(config.csv_output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write AnythingLLM metrics
            writer.writerow(["AnythingLLM Metrics"])
            writer.writerow(["User ID", "Iteration", "Status", "Latency (ms)", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Output TPS (tokens/s)", "Duration (s)"])
            for result in results:
                metrics = result.get('metrics', {})
                writer.writerow([
                    result['user_id'],
                    result['iteration'],
                    result['status'],
                    result['latency_ms'],
                    metrics.get('prompt_tokens', ''),
                    metrics.get('completion_tokens', ''),
                    metrics.get('total_tokens', ''),
                    metrics.get('outputTps', ''),
                    metrics.get('duration', '')
                ])
            
            # Write NVIDIA metrics
            writer.writerow([])
            writer.writerow(["NVIDIA Metrics"])
            writer.writerow(["Timestamp", "GPU Utilization (%)", "Memory Utilization (%)", "Memory Used (MiB)", "Memory Total (MiB)"])
            for metric in nvidia_metrics:
                writer.writerow([
                    metric['timestamp'],
                    metric['gpu_utilization'],
                    metric['memory_utilization'],
                    metric['memory_used_mib'],
                    metric['memory_total_mib']
                ])
            
            # Write llama.cpp metrics for each container
            for container, metrics in llama_metrics.items():
                writer.writerow([])
                writer.writerow([f"llama.cpp Metrics ({container})"])
                writer.writerow(["Timestamp", "Type", "Time (ms)", "Tokens", "TPS (tokens/s)"])
                for metric in metrics:
                    writer.writerow([
                        metric['timestamp'],
                        metric['type'],
                        metric.get('time_ms', ''),
                        metric.get('tokens', ''),
                        metric.get('tps', '')
                    ])
        logger.info(f"Metrics saved to CSV: {config.csv_output_file}")
    except PermissionError as e:
        logger.error(f"Failed to write metrics to CSV at {config.csv_output_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error saving metrics to CSV: {str(e)}")

def compute_statistics(values: List[float]) -> Dict[str, Any]:
    """
    Compute statistics for a list of values: mean, std dev, CoV, 95% CI, reliability.
    
    Args:
        values (List[float]): List of values to analyze.
    
    Returns:
        Dict[str, Any]: Dictionary containing mean, std dev, CoV, 95% CI, and reliability.
    """
    if not values:
        return {
            "mean": 0.0,
            "std_dev": 0.0,
            "cov": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "reliable": "No"
        }
    
    n = len(values)
    mean_val = mean(values)
    std_dev = stdev(values) if n > 1 else 0.0
    cov = (std_dev / mean_val * 100) if mean_val != 0 else 0.0
    
    # Calculate 95% confidence interval: mean ± 1.96 * (std_dev / sqrt(n))
    if n > 1:
        margin_error = 1.96 * (std_dev / math.sqrt(n))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
    else:
        ci_lower = mean_val
        ci_upper = mean_val
    
    # Reliability: CoV < 10%
    reliable = "Yes" if cov < 10 else "No"
    
    return {
        "mean": mean_val,
        "std_dev": std_dev,
        "cov": cov,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "reliable": reliable
    }

def save_results_to_markdown(
    config: Config,
    response_latency: List[float],
    nvidia_metrics: List[Dict[str, Any]],
    llama_metrics: Dict[str, List[Dict[str, Any]]]
) -> None:
    """
    Save performance metrics to a Markdown file in the log subdirectory.
    
    Args:
        config (Config): Configuration object with file paths.
        response_latency (List[float]): List of response latencies from AnythingLLM.
        nvidia_metrics (List[Dict[str, Any]]): List of NVIDIA metrics.
        llama_metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of llama.cpp metrics for each container.
    """
    try:
        # Compute statistics for each metric
        response_latency_stats = compute_statistics(response_latency)
        gpu_utilization = [m['gpu_utilization'] for m in nvidia_metrics]
        gpu_stats = compute_statistics(gpu_utilization)
        memory_utilization = [m['memory_utilization'] for m in nvidia_metrics]
        memory_stats = compute_statistics(memory_utilization)
        
        # Use metrics from poc-llama-cpp-mistral-1 (active container)
        mistral_metrics = llama_metrics.get('poc-llama-cpp-mistral-1', [])
        prompt_eval_tps = [m['tps'] for m in mistral_metrics if m['type'] == "prompt_eval"]
        prompt_eval_tps_stats = compute_statistics(prompt_eval_tps)
        generation_tps = [m['tps'] for m in mistral_metrics if m['type'] == "eval"]
        generation_tps_stats = compute_statistics(generation_tps)
        ttft = [m['time_ms'] for m in mistral_metrics if m['type'] == "prompt_eval"]
        ttft_stats = compute_statistics(ttft)
        
        # Format the Markdown table
        table_rows = [
            "| Metric                 | Mean Value         | Standard Deviation | Coefficient of Variation (%) | 95% Confidence Interval   | Reliable? |",
            "|------------------------|--------------------|--------------------|------------------------------|---------------------------|-----------|",
            f"| Prompt Eval TPS        | {prompt_eval_tps_stats['mean']:.2f} tokens/s | {prompt_eval_tps_stats['std_dev']:.2f} | {prompt_eval_tps_stats['cov']:.1f} | [{prompt_eval_tps_stats['ci_lower']:.2f}, {prompt_eval_tps_stats['ci_upper']:.2f}] | {prompt_eval_tps_stats['reliable']} |",
            f"| Generation TPS         | {generation_tps_stats['mean']:.2f} tokens/s | {generation_tps_stats['std_dev']:.2f} | {generation_tps_stats['cov']:.1f} | [{generation_tps_stats['ci_lower']:.2f}, {generation_tps_stats['ci_upper']:.2f}] | {generation_tps_stats['reliable']} |",
            f"| Response Latency (TTFT)| {ttft_stats['mean']:.2f} ms | {ttft_stats['std_dev']:.2f} | {ttft_stats['cov']:.1f} | [{ttft_stats['ci_lower']:.2f}, {ttft_stats['ci_upper']:.2f}] | {ttft_stats['reliable']} |",
            f"| Response Latency (Total)| {response_latency_stats['mean']:.2f} ms | {response_latency_stats['std_dev']:.2f} | {response_latency_stats['cov']:.1f} | [{response_latency_stats['ci_lower']:.2f}, {response_latency_stats['ci_upper']:.2f}] | {response_latency_stats['reliable']} |",
            f"| GPU Utilization (%)    | {gpu_stats['mean']:.2f}% | {gpu_stats['std_dev']:.2f} | {gpu_stats['cov']:.1f} | [{gpu_stats['ci_lower']:.2f}, {gpu_stats['ci_upper']:.2f}] | {gpu_stats['reliable']} |",
            f"| Memory Utilization (%) | {memory_stats['mean']:.2f}% | {memory_stats['std_dev']:.2f} | {memory_stats['cov']:.1f} | [{memory_stats['ci_lower']:.2f}, {memory_stats['ci_upper']:.2f}] | {memory_stats['reliable']} |",
        ]
        
        # Write to results.md
        with open(config.markdown_output_file, 'w') as f:
            f.write("\n".join(table_rows) + "\n")
        logger.info(f"Results saved to Markdown: {config.markdown_output_file}")
    except PermissionError as e:
        logger.error(f"Failed to write results to Markdown at {config.markdown_output_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error saving results to Markdown: {str(e)}")

def main() -> None:
    """
    Main function to execute the performance test with multiple users and log the results.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config()
    if args.log_level:
        config.log_level = args.log_level
    if args.num_users:
        config.num_users = args.num_users
    if args.num_iterations:
        config.num_iterations = args.num_iterations
    config.validate()
    
    # Setup logging
    setup_logging(config)
    
    logger.info(f"Starting performance test with {config.num_users} users, {config.num_iterations} iterations per user")
    
    # Start capturing NVIDIA metrics and llama.cpp logs in background threads
    stop_event = threading.Event()
    
    nvidia_thread = threading.Thread(target=capture_nvidia_smi_metrics, args=(config, stop_event))
    nvidia_thread.start()
    
    llama_cpp_thread = threading.Thread(target=capture_llama_cpp_logs, args=(config, stop_event))
    llama_cpp_thread.start()
    
    # Use a session for connection reuse
    results = []
    threads = []
    with Session() as session:
        try:
            # Start threads for each user
            for user_id in range(config.num_users):
                thread = threading.Thread(target=run_user, args=(config, session, user_id + 1, results))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
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
    response_latency = []
    if successful_results:
        response_latency = [r['latency_ms'] for r in successful_results]
        prompt_tokens = [r['metrics']['prompt_tokens'] for r in successful_results if 'metrics' in r and 'prompt_tokens' in r['metrics']]
        completion_tokens = [r['metrics']['completion_tokens'] for r in successful_results if 'metrics' in r and 'completion_tokens' in r['metrics']]
        total_tokens = [r['metrics']['total_tokens'] for r in successful_results if 'metrics' in r and 'total_tokens' in r['metrics']]
        output_tps = [r['metrics']['outputTps'] for r in successful_results if 'metrics' in r and 'outputTps' in r['metrics']]
        durations = [r['metrics']['duration'] for r in successful_results if 'metrics' in r and 'duration' in r['metrics']]
        
        # Calculate percentiles for latency
        latency_percentiles = quantiles(response_latency, n=10) if len(response_latency) >= 10 else response_latency
        
        logger.info("\nPerformance Metrics:")
        logger.info(f"Response Latency (Total): {mean(response_latency):.2f} ms (Std Dev: {stdev(response_latency) if len(response_latency) > 1 else 0:.2f} ms)")
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
    
    # Save results to Markdown
    save_results_to_markdown(config, response_latency, nvidia_metrics, llama_metrics)
    
    # Save metrics to CSV
    save_metrics_to_csv(config, results, nvidia_metrics, llama_metrics)

if __name__ == "__main__":
    main()