# API Example Readme

This folder contains examples to query Anything LLM using Bash and Python scripts. This approach was used during performance testing. 

However, where document RAG or other add-on features are not needed, it would be more efficient to query llama-server directly as in the following example.
```
curl -X POST http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Please generate a C sharp program to print the first 100 numbers in the fibonacci sequence", "max_tokens": 500}'
```