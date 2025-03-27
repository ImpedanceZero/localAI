import requests
import os
import time
import logging
import json
import sys
import argparse
from typing import Dict, Any, Optional, Literal
from logging.handlers import RotatingFileHandler
from requests.exceptions import RequestException, Timeout, HTTPError
from requests.sessions import Session
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, ValidationError

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
        self.log_file: str = os.getenv("ANYTHING_LLM_LOG_FILE", "anything_llm_test.log")
        self.log_max_bytes: int = int(os.getenv("ANYTHING_LLM_LOG_MAX_BYTES", 5 * 1024 * 1024))
        self.log_backup_count: int = int(os.getenv("ANYTHING_LLM_LOG_BACKUP_COUNT", 3))
        self.request_timeout: int = int(os.getenv("ANYTHING_LLM_REQUEST_TIMEOUT", 30))
        self.api_key: str = os.getenv("ANYTHING_LLM_API_KEY", "")
        self.chat_endpoint: str = f"{self.base_url}/api/v1/workspace/{self.test_workspace}/chat"

    def validate(self) -> None:
        if not self.api_key:
            raise AnythingLLMConfigError("API key not found. Please set the ANYTHING_LLM_API_KEY environment variable.")

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
        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Timeout, HTTPError)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying request (attempt {retry_state.attempt_number}/3) after {retry_state.idle_for:.2f}s..."
    )
)
def submit_query(prompt: str, session: Session, config: Config) -> Dict[str, Any]:
    """
    Submit a single query to AnythingLLM in the specified workspace and return the response.
    
    Args:
        prompt (str): The query to submit.
        session (Session): HTTP session for connection reuse.
        config (Config): Configuration object with API settings.
    
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
        logger.debug(f"Sending request to {config.chat_endpoint}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
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
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        logger.debug(f"Raw Response: {truncate_string(response.text)}")
        
        # Check Content-Type to ensure it's JSON
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            logger.warning(f"Response Content-Type is not JSON: {content_type}")
            raise AnythingLLMRequestError(f"Expected JSON response, got Content-Type: {content_type}")
        
        # Parse response as JSON
        response_data = response.json()
        
        # Validate response structure with Pydantic
        validated_response = AnythingLLMResponse(**response_data)
        
        logger.info("Request successful")
        return {
            "status": "Success",
            "latency_ms": latency,
            "response": validated_response.dict()
        }
    
    except Timeout as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Request timed out after {config.request_timeout} seconds: {str(e)}")
        raise AnythingLLMRequestError(f"Request timed out after {config.request_timeout} seconds: {str(e)}")
    except HTTPError as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"HTTP error occurred: {str(e)}")
        raise AnythingLLMRequestError(f"HTTP error: {str(e)}")
    except RequestException as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Request failed with exception: {str(e)}")
        raise AnythingLLMRequestError(f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"Failed to parse JSON response: {str(e)}")
        raise AnythingLLMRequestError(f"Failed to parse JSON response: {str(e)}")
    except ValidationError as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"Response validation failed: {str(e)}")
        raise AnythingLLMRequestError(f"Response validation failed: {str(e)}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for log level configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test script for AnythingLLM API.")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides ANYTHING_LLM_LOG_LEVEL environment variable)"
    )
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the test query and log the results.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config()
    if args.log_level:
        config.log_level = args.log_level
    config.validate()
    
    # Setup logging
    setup_logging(config)
    
    logger.info(f"Starting test with workspace '{config.test_workspace}' and prompt: {config.test_prompt}")
    
    # Use a session for connection reuse
    with Session() as session:
        try:
            # Submit the query
            result = submit_query(config.test_prompt, session, config)
            
            # Log the results
            logger.info("Test Results:")
            logger.info(f"Status: {result['status']}")
            logger.info(f"Latency: {result['latency_ms']:.2f} ms")
            
            # Format the response for human-readable output
            if result['status'] == "Success":
                # Pretty-print the JSON response with indentation
                formatted_response = json.dumps(result['response'], indent=2)
                logger.info("Response:")
                logger.info(formatted_response)
            else:
                logger.info(f"Response: {result['response']}")
        except AnythingLLMRequestError as e:
            logger.error(f"Failed to submit query: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()