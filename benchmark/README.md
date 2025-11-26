# AWS Bedrock LLM Benchmarking Framework

A comprehensive benchmarking framework for testing AWS Bedrock LLMs with concurrent load testing, performance metrics, error tracking, and cost analysis across multiple prompts.

## Quick Start

```bash
# Run with all configured prompts and models
uv run python src/benchmark/main.py --number-of-calls 10

# Test specific model with all prompts  
uv run python src/benchmark/main.py --specific-model "Claude-3-Haiku" --number-of-calls 5

# Test specific prompt with all models
uv run python src/benchmark/main.py --specific-prompt "cloud_computing" --number-of-calls 5

# Use custom prompt instead of configured prompts
uv run python src/benchmark/main.py --prompt "What is machine learning?" --number-of-calls 3
```

## Prerequisites

1. **AWS Credentials**: Configure AWS credentials with Bedrock access:
   ```bash
   aws configure
   # OR set environment variables:
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_DEFAULT_REGION=eu-central-1
   ```

2. **Dependencies**: Install using uv:
   ```bash
   uv sync
   ```

## Configuration

### Models (`config/models.json`)
Configure which Bedrock models to test:
```json
{
  "models": [
    {
      "name": "Claude-3-Haiku",
      "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  ]
}
```

### Prompts (`config/prompts.json`)
Configure test prompts (used when no custom prompt provided):
```json
{
  "prompts": [
    {
      "name": "cloud_computing",
      "description": "Cloud computing benefits analysis", 
      "prompt": "What are the key benefits of cloud computing?"
    }
  ]
}
```

### Pricing (`config/pricing.csv`)
Model pricing for cost calculations:
```csv
model_name,input_cost_per_1k_tokens,output_cost_per_1k_tokens
Claude-3-Haiku,0.00025,0.00125
```

## CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `--number-of-calls` | Calls per model per prompt | `--number-of-calls 50` |
| `--specific-model` | Test only one model | `--specific-model "Claude-3-Haiku"` |
| `--specific-prompt` | Test only one prompt | `--specific-prompt "cloud_computing"` |
| `--prompt` | Custom prompt (overrides config) | `--prompt "Explain AI"` |
| `--region` | AWS region | `--region us-east-1` |

## Output Structure

When running with multiple prompts, results are organized as:

```
outputs/YYYYMMDD_HHMMSS/
├── results_overall.csv          # Combined results from all prompts
├── cloud_computing/             # Results for first prompt
│   ├── prompt_info.json        # Prompt configuration
│   ├── prompt_results.csv      # Results for this prompt
│   ├── answers_Claude-3-Haiku.json
│   └── answers_Claude-3-Sonnet.json
├── data_science/               # Results for second prompt
│   ├── prompt_info.json
│   ├── prompt_results.csv
│   ├── answers_Claude-3-Haiku.json
│   └── answers_Claude-3-Sonnet.json
└── software_architecture/      # Results for third prompt
    ├── prompt_info.json
    ├── prompt_results.csv  
    ├── answers_Claude-3-Haiku.json
    └── answers_Claude-3-Sonnet.json
```

## Example Usage

### Test All Prompts with All Models
```bash
uv run python src/benchmark/main.py --number-of-calls 10
```
This runs 10 calls × 3 prompts × 2 models = 60 total API calls

### Test Specific Combinations
```bash
# Only test cloud computing prompt with Haiku model
uv run python src/benchmark/main.py \
  --specific-prompt "cloud_computing" \
  --specific-model "Claude-3-Haiku" \
  --number-of-calls 5
```

### Use Custom Prompt
```bash
# Test custom prompt with all models (ignores prompt config)
uv run python src/benchmark/main.py \
  --prompt "Explain quantum computing in simple terms" \
  --number-of-calls 3
```

## Output Files

### `results_overall.csv`
Combined metrics across all prompts and models with columns:
- model_name, total_calls, successful_calls, success_rate
- avg_response_time, min_response_time, max_response_time  
- total_input_tokens, total_output_tokens, total_cost
- error counts by type (timeout, rate_limit, service, etc.)

### `prompt_results.csv` 
Same format but filtered to specific prompt results

### `answers_[model].json`
Detailed response data including:
```json
{
  "model_name": "Claude-3-Haiku",
  "timestamp": "2024-11-26T10:30:00Z",
  "prompt_info": {
    "name": "cloud_computing",
    "description": "...",
    "prompt": "..."
  },
  "responses": [
    {
      "call_id": 1,
      "success": true,
      "response_time": 0.85,
      "input_tokens": 25,
      "output_tokens": 150,
      "response": "Generated text...",
      "error": null
    }
  ]
}
```

### `prompt_info.json`
Stores the exact prompt configuration used for each test

## Cost Management

- Check pricing in `config/pricing.csv` 
- Start with low `--number-of-calls` for testing
- Monitor costs in output summaries
- Tool warns for >1000 calls and requires confirmation

## Troubleshooting

**Import Error**: Ensure you're in the project root directory

**AWS Credentials**: Verify with `aws sts get-caller-identity`

**Rate Limits**: Reduce `--number-of-calls` or test fewer models/prompts

**Region Issues**: Ensure Bedrock is available in your region