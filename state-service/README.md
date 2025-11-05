# CodeAssistant State Service

A FastAPI-based service that provides AI-powered text processing using Ollama with the `qwen2.5-coder:0.5b-base` model and ASM (Action Selection Model) for intelligent code completion.

## Features

- **Text Processing**: Preprocesses input text and generates AI responses using FIM (Fill-in-Middle) approach
- **Action Selection**: Uses ASM model to determine the appropriate action for code completion
- **Timestamp Support**: Handles timestamp information for context
- **Code Extraction**: Automatically extracts code blocks from responses
- **Health Monitoring**: Provides health checks for Ollama and model availability
- **Error Handling**: Comprehensive error handling and validation
- **CORS Support**: Cross-origin resource sharing enabled for web applications

## Architecture

The service consists of several key components:

- **`server.py`**: Main FastAPI application with lifespan management
- **`ollama_client.py`**: Ollama API client for model interactions
- **`asm_client.py`**: Action Selection Model client (currently mocked)
- **`preprocessor.py`**: Text preprocessing logic for FIM prompts
- **`postprocessor.py`**: Response postprocessing logic
- **`datatypes.py`**: Data structures including ActionIndex enum and InferenceRequest
- **`config.py`**: Configuration management with environment variable support
- **`utils.py`**: Utility functions for response formatting

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is Running**:
   Make sure the Ollama container is running with the `qwen2.5-coder:0.5b-base` model available.

3. **Start the Service**:
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### Root
- **GET** `/` - Service information and version

### Health Check
- **GET** `/health` - Check service and Ollama health

### Inference
- **POST** `/inference` - Process text and generate AI response with action selection

### Model Information
- **GET** `/model-info` - Get information about the current model

## Usage Examples

### Basic Inference Request
```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "def calculate_area(ra",
    "author_attribution": "user",
    "timestamp": "2024-01-15T10:30:00Z",
    "context": {"file_type": "python"}
  }'
```

### Testing
Run the test client to verify the service:
```bash
python test_client.py
```

Run model comparison tests:
```bash
python test_model_comparison.py
```

## Configuration

The service can be configured using environment variables:

- `OLLAMA_BASE_URL`: Ollama API base URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model to use (default: qwen2.5-coder:0.5b-base)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Enable debug mode (default: false)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 300)

We will let the state service handle generation kwargs like since these will vary per action 
- `MAX_TOKENS`: Default max tokens (default: 2048)
- `TEMPERATURE`: Default temperature (default: 0.7)
- `TOP_P`: Default top-p value (default: 0.9)

## Action Types

The service supports various action types for code completion:

- `NO_OP`: No operation
- `WRITE_PARTIAL_LINE`: Complete partial line
- `WRITE_SINGLE_LINE`: Write single line
- `WRITE_MULTI_LINE`: Write multiple lines
- `COMMENT_PARTIAL_LINE`: Comment partial line
- `COMMENT_SINGLE_LINE`: Comment single line
- `COMMENT_MULTI_LINE`: Comment multiple lines
- `EDIT`: Edit existing code
- `DELETE`: Delete code

## Model Selection

The service uses the `qwen2.5-coder:0.5b-base` model by default because it provides better FIM (Fill-in-Middle) completions compared to other variants. The base model is more focused and sticks to the FIM template without generating excessive text.

## Quick Start

### Option 1: Docker

```bash
# Build the Docker image
docker build -t codeassist-state-service:latest .

# Run the container
docker run -d -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 --name state-service codeassist-state-service:latest

# Check container status
docker ps

# View logs
docker logs state-service
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the service
python main.py

# Test the service
python test_client.py

# Run model comparison tests
python test_model_comparison.py
```

The service will be available at `http://localhost:8000` with automatic health checks and model availability verification on startup.
