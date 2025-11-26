import json
import os
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd


def load_models_config(config_path: str = "config/models.json") -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Models configuration file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in models configuration: {e}")


def load_pricing_data(models_config_path: str = "config/models.json") -> Dict[str, Dict[str, float]]:
    try:
        models_config = load_models_config(models_config_path)
        pricing_data = {}
        
        for model in models_config["models"]:
            model_name = model["name"]
            pricing_data[model_name] = {
                "input_cost_per_1k_tokens": float(model.get("input_cost_per_1k_tokens", 0.0)),
                "output_cost_per_1k_tokens": float(model.get("output_cost_per_1k_tokens", 0.0))
            }
        return pricing_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Models configuration file not found at {models_config_path}")
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid models configuration format: {e}")


def load_prompts_config(prompts_path: str = "config/prompts.json") -> Dict[str, Any]:
    try:
        with open(prompts_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompts configuration file not found at {prompts_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in prompts configuration: {e}")


def create_output_directory(base_path: str = "outputs") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_prompt_output_directory(base_output_dir: str, prompt_name: str) -> str:
    prompt_dir = os.path.join(base_output_dir, prompt_name)
    os.makedirs(prompt_dir, exist_ok=True)
    return prompt_dir


def save_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    if not results:
        return
    
    csv_data = []
    for result in results:
        metrics = result["metrics"]
        csv_row = {
            "model_name": result["model_name"],
            "total_calls": metrics["total_calls"],
            "successful_calls": metrics["successful_calls"],
            "failed_calls": metrics["failed_calls"],
            "success_rate": round(metrics["success_rate"], 2),
            "error_rate": round(metrics["error_rate"], 2),
            "throughput": round(metrics["throughput"], 2),
            "avg_response_time": round(metrics["avg_response_time"], 3),
            "min_response_time": round(metrics["min_response_time"], 3),
            "max_response_time": round(metrics["max_response_time"], 3),
            "total_input_tokens": metrics["total_input_tokens"],
            "total_output_tokens": metrics["total_output_tokens"],
            "total_cost": round(metrics["total_cost"], 6),
            "cost_per_call": round(metrics["cost_per_call"], 6),
            "errors_timeout": metrics["errors_timeout"],
            "errors_rate_limit": metrics["errors_rate_limit"],
            "errors_service": metrics["errors_service"],
            "errors_auth": metrics["errors_auth"],
            "errors_validation": metrics["errors_validation"],
            "errors_exception": metrics["errors_exception"]
        }
        csv_data.append(csv_row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)


def save_answers_json(result: Dict[str, Any], output_path: str) -> None:
    output_data = {
        "model_name": result["model_name"],
        "timestamp": result["timestamp"],
        "prompt_info": result.get("prompt_info", {}),
        "responses": result["responses"]
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def save_prompt_info_json(prompt_config: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(prompt_config, f, indent=2)


def aggregate_results_by_model(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate results by model name across all prompts.
    Combines metrics from multiple prompt runs into overall model performance.
    """
    model_aggregates = {}
    
    for result in all_results:
        model_name = result["model_name"]
        metrics = result["metrics"]
        
        if model_name not in model_aggregates:
            model_aggregates[model_name] = {
                "model_name": model_name,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "response_times": [],
                "errors_timeout": 0,
                "errors_rate_limit": 0,
                "errors_service": 0,
                "errors_auth": 0,
                "errors_validation": 0,
                "errors_exception": 0,
            }
        
        agg = model_aggregates[model_name]
        
        # Sum up counts and totals
        agg["total_calls"] += metrics["total_calls"]
        agg["successful_calls"] += metrics["successful_calls"]
        agg["failed_calls"] += metrics["failed_calls"]
        agg["total_input_tokens"] += metrics["total_input_tokens"]
        agg["total_output_tokens"] += metrics["total_output_tokens"]
        agg["total_cost"] += metrics["total_cost"]
        
        # Collect response times for later aggregation
        if "responses" in result:
            for response in result["responses"]:
                if response["success"]:
                    agg["response_times"].append(response["response_time"])
        
        # Sum error counts
        agg["errors_timeout"] += metrics["errors_timeout"]
        agg["errors_rate_limit"] += metrics["errors_rate_limit"]
        agg["errors_service"] += metrics["errors_service"]
        agg["errors_auth"] += metrics["errors_auth"]
        agg["errors_validation"] += metrics["errors_validation"]
        agg["errors_exception"] += metrics["errors_exception"]
    
    # Calculate final aggregated metrics
    aggregated_results = []
    for model_name, agg in model_aggregates.items():
        # Calculate rates and averages
        success_rate = (agg["successful_calls"] / agg["total_calls"]) * 100 if agg["total_calls"] > 0 else 0
        error_rate = (agg["failed_calls"] / agg["total_calls"]) * 100 if agg["total_calls"] > 0 else 0
        cost_per_call = agg["total_cost"] / agg["successful_calls"] if agg["successful_calls"] > 0 else 0
        
        # Calculate response time metrics
        if agg["response_times"]:
            avg_response_time = sum(agg["response_times"]) / len(agg["response_times"])
            min_response_time = min(agg["response_times"])
            max_response_time = max(agg["response_times"])
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        aggregated_results.append({
            "model_name": model_name,
            "total_calls": agg["total_calls"],
            "successful_calls": agg["successful_calls"],
            "failed_calls": agg["failed_calls"],
            "success_rate": success_rate,
            "error_rate": error_rate,
            "throughput": 0,  # Cannot aggregate throughput meaningfully across different time periods
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "total_input_tokens": agg["total_input_tokens"],
            "total_output_tokens": agg["total_output_tokens"],
            "total_cost": agg["total_cost"],
            "cost_per_call": cost_per_call,
            "errors_timeout": agg["errors_timeout"],
            "errors_rate_limit": agg["errors_rate_limit"],
            "errors_service": agg["errors_service"],
            "errors_auth": agg["errors_auth"],
            "errors_validation": agg["errors_validation"],
            "errors_exception": agg["errors_exception"]
        })
    
    return aggregated_results


def save_aggregated_results_csv(aggregated_results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save aggregated results to CSV with the same format as individual results.
    """
    if not aggregated_results:
        return
    
    csv_data = []
    for result in aggregated_results:
        csv_row = {
            "model_name": result["model_name"],
            "total_calls": result["total_calls"],
            "successful_calls": result["successful_calls"],
            "failed_calls": result["failed_calls"],
            "success_rate": round(result["success_rate"], 2),
            "error_rate": round(result["error_rate"], 2),
            "throughput": "N/A",  # Cannot aggregate meaningfully
            "avg_response_time": round(result["avg_response_time"], 3),
            "min_response_time": round(result["min_response_time"], 3),
            "max_response_time": round(result["max_response_time"], 3),
            "total_input_tokens": result["total_input_tokens"],
            "total_output_tokens": result["total_output_tokens"],
            "total_cost": round(result["total_cost"], 6),
            "cost_per_call": round(result["cost_per_call"], 6),
            "errors_timeout": result["errors_timeout"],
            "errors_rate_limit": result["errors_rate_limit"],
            "errors_service": result["errors_service"],
            "errors_auth": result["errors_auth"],
            "errors_validation": result["errors_validation"],
            "errors_exception": result["errors_exception"]
        }
        csv_data.append(csv_row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)


def validate_aws_credentials() -> bool:
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_cost(cost: float) -> str:
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    
    # Header
    print(f"{'Model':<20} {'Calls':<8} {'Success%':<9} {'Avg Time':<10} {'Throughput':<12} {'Total Cost':<12} {'Errors':<8}")
    print("-" * 100)
    
    # Data rows
    for result in results:
        metrics = result["metrics"]
        model_name = result["model_name"][:19]  # Truncate if too long
        
        print(f"{model_name:<20} "
              f"{metrics['total_calls']:<8} "
              f"{metrics['success_rate']:<8.1f}% "
              f"{metrics['avg_response_time']:<9.3f}s "
              f"{metrics['throughput']:<11.2f}/s "
              f"{format_cost(metrics['total_cost']):<12} "
              f"{metrics['failed_calls']:<8}")
    
    print("="*100)


def get_default_prompt() -> str:
    return (
        "You are a helpful AI assistant. Please provide a comprehensive response to the following question: "
        "What are the key benefits of using cloud computing for modern businesses? "
        "Include at least 5 specific advantages and explain each one briefly."
    )