# AWS Bedrock LLM Benchmarking Framework

## Overview
A comprehensive benchmarking framework for testing AWS Bedrock LLMs with concurrent load testing, performance metrics, error tracking, and cost analysis.

## Project Structure
```
benchmark/
├── pyproject.toml          # uv project configuration
├── instructions.md         # This documentation
├── src/
│   └── benchmark/
│       ├── __init__.py
│       ├── main.py         # CLI entry point with argument parsing
│       ├── bedrock_client.py # Bedrock API wrapper with boto3
│       ├── benchmarker.py  # Core benchmarking logic
│       └── utils.py        # Helper functions (file I/O, formatting)
├── config/
│   ├── models.json         # Model configurations (easily editable)
│   └── pricing.csv         # Model pricing data for cost calculations
└── outputs/               # Generated benchmark results
    └── YYYYMMDD_HHMMSS/   # Timestamped run directory
        ├── results.csv    # Aggregated metrics and statistics
        ├── answers_model1.json  # Individual responses per model
        ├── answers_model2.json
        └── answers_modelN.json
```

## Features

### Core Functionality
- **Multi-Model Testing**: Test multiple Bedrock models sequentially
- **Async Concurrent Calls**: Execute hundreds of calls simultaneously per model
- **Comprehensive Metrics**: Response times, token usage, costs, error rates
- **Error Tracking**: Success rates, error categorization, failure analysis
- **Flexible Configuration**: JSON-based model management
- **CLI Interface**: Optional parameters for targeted testing

### Metrics Collected
1. **Performance Metrics**
   - Response time: average, minimum, maximum
   - Throughput: calls per second
   - Success rate percentage

2. **Token & Cost Analysis**
   - Input tokens consumed
   - Output tokens generated
   - Cost per model (using pricing.csv)
   - Cost per successful call

3. **Error Tracking**
   - Total errors encountered
   - Error types (timeout, rate limit, service error, etc.)
   - Failed call percentage
   - Error distribution analysis

### Output Structure
- **Timestamped Directories**: Each run creates `outputs/YYYYMMDD_HHMMSS/`
- **Aggregated Results**: `results.csv` with all metrics per model
- **Detailed Responses**: `answers_[modelName].json` files containing all individual responses
- **Error Logs**: Detailed error information for debugging

## Configuration

### Models Configuration (`config/models.json`)
```json
{
  "models": [
    {
      "name": "Claude-3-Haiku",
      "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
      "max_tokens": 4096,
      "temperature": 0.7
    },
    {
      "name": "Claude-3-Sonnet", 
      "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  ]
}
```

### Pricing Data (`config/pricing.csv`)
```csv
model_name,input_cost_per_1k_tokens,output_cost_per_1k_tokens
Claude-3-Haiku,0.00025,0.00125
Claude-3-Sonnet,0.003,0.015
```

## Usage

### Basic Usage
```bash
# Run benchmark on all configured models with default settings
uv run src/benchmark/main.py

# Test specific model only
uv run src/benchmark/main.py --specific-model "Claude-3-Haiku"

# Control number of concurrent calls
uv run src/benchmark/main.py --number-of-calls 50

# Combined parameters
uv run src/benchmark/main.py --specific-model "Claude-3-Sonnet" --number-of-calls 200
```

### CLI Parameters
- `--specific-model`: Test only one specific model (use model name from config)
- `--number-of-calls`: Number of asynchronous calls per model (default: 100)
- `--prompt`: Custom prompt for testing (optional, uses default if not provided)

## Implementation Components

### 1. Bedrock Client (`bedrock_client.py`)
- boto3 Bedrock Runtime client wrapper
- Configured for eu-central-1 region
- Async API call handling
- Error handling and retry logic
- Token counting and response parsing

### 2. Benchmarker (`benchmarker.py`)
- Core benchmarking engine
- Async execution manager
- Performance metrics collection
- Error tracking and categorization
- Results aggregation and statistics calculation

### 3. CLI Interface (`main.py`)
- Argument parsing for CLI parameters
- Progress tracking and status updates
- Results export coordination
- Logging and error reporting

### 4. Utilities (`utils.py`)
- File I/O operations
- CSV and JSON formatting
- Timestamp generation
- Directory management
- Data validation helpers

## Technical Requirements
- **Python**: 3.8+
- **Package Manager**: uv
- **AWS SDK**: boto3
- **Region**: eu-central-1
- **Dependencies**: asyncio, aiofiles, pandas, click

## Error Handling
- **API Errors**: Timeout, rate limiting, service unavailability
- **Network Issues**: Connection failures, DNS resolution
- **Authentication**: AWS credential validation
- **Data Validation**: Model configuration, pricing data
- **Graceful Degradation**: Continue testing other models on individual failures

## Output Examples

### Results CSV Structure
```csv
model_name,total_calls,successful_calls,error_rate,avg_response_time,min_response_time,max_response_time,total_input_tokens,total_output_tokens,total_cost,cost_per_call,errors_timeout,errors_rate_limit,errors_service
```

### Answer JSON Structure
```json
{
  "model_name": "Claude-3-Haiku",
  "timestamp": "2024-11-26T10:30:00Z",
  "responses": [
    {
      "call_id": 1,
      "success": true,
      "response_time": 0.85,
      "input_tokens": 25,
      "output_tokens": 150,
      "response": "Generated text response...",
      "error": null
    }
  ]
}
```

## Future Enhancements
- Support for custom prompts per model
- Batch processing for large-scale testing
- Real-time monitoring dashboard
- Model comparison visualization
- Historical performance tracking