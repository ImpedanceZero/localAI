#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --base-url <base_url> --workspace <workspace_slug> --prompt <prompt>"
    echo "Example: $0 --base-url http://llm.blue.local --workspace docexplorer-document-rag-chat --prompt 'How many calories are in a Quiznos Classic Italian sub?'"
    exit 1
}

# Check if API key is set in environment variables
if [ -z "$ANYTHING_LLM_API_KEY" ]; then
    echo "Error: API key not found. Please set the ANYTHING_LLM_API_KEY environment variable."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq to parse JSON responses."
    echo "On WSL/Ubuntu, you can install it with: sudo apt-get install jq"
    exit 1
fi

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base-url) BASE_URL="$2"; shift ;;
        --workspace) WORKSPACE="$2"; shift ;;
        --prompt) PROMPT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Validate required parameters
if [ -z "$BASE_URL" ] || [ -z "$WORKSPACE" ] || [ -z "$PROMPT" ]; then
    echo "Error: Missing required parameters."
    usage
fi

# Construct the endpoint
ENDPOINT="$BASE_URL/api/v1/workspace/$WORKSPACE/chat"

# Log the request details (excluding API key)
echo "Sending request to $ENDPOINT"
echo "Prompt: $PROMPT"

# Send the chat request using curl and pipe the response to jq for formatting
curl -s -X POST "$ENDPOINT" \
     -H "Authorization: Bearer $ANYTHING_LLM_API_KEY" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d "{\"message\": \"$PROMPT\", \"mode\": \"chat\"}" | jq .

# Add a newline for readability
echo ""