import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import statistics
from bedrock_client import BedrockClient
from utils import load_models_config, load_pricing_data, load_prompts_config


class Benchmarker:
    def __init__(self, region_name: str = "eu-central-1"):
        self.client = BedrockClient(region_name)
        self.models_config = load_models_config()
        self.pricing_data = load_pricing_data()
        self.prompts_config = load_prompts_config()
        
    async def benchmark_model(
        self,
        model_config: Dict[str, Any],
        prompt: str,
        prompt_info: Dict[str, Any],
        number_of_calls: int = 100
    ) -> Dict[str, Any]:
        model_name = model_config["name"]
        model_id = model_config["model_id"]
        max_tokens = model_config.get("max_tokens", 4096)
        temperature = model_config.get("temperature", 0.7)
        
        print(f"Starting benchmark for {model_name} with {number_of_calls} calls...")
        
        # Create tasks for concurrent execution
        tasks = []
        for call_id in range(number_of_calls):
            task = asyncio.create_task(
                self._make_call_with_id(
                    call_id + 1,
                    model_id,
                    prompt,
                    max_tokens,
                    temperature
                )
            )
            tasks.append(task)
        
        # Execute all calls concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({
                    "success": False,
                    "error": {
                        "type": "exception",
                        "code": "TaskException", 
                        "message": str(result)
                    }
                })
            elif result["success"]:
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            model_name,
            successful_results,
            failed_results,
            total_time,
            number_of_calls
        )
        
        # Prepare detailed responses for JSON output
        detailed_responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                detailed_responses.append({
                    "call_id": i + 1,
                    "success": False,
                    "response_time": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "response": "",
                    "error": {
                        "type": "exception",
                        "code": "TaskException",
                        "message": str(result)
                    }
                })
            else:
                detailed_responses.append({
                    "call_id": i + 1,
                    "success": result["success"],
                    "response_time": result["response_time"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "response": result["response"],
                    "error": result["error"]
                })
        
        return {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "prompt_info": prompt_info,
            "metrics": metrics,
            "responses": detailed_responses
        }
    
    async def _make_call_with_id(
        self,
        call_id: int,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        result = await self.client.invoke_model(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        result["call_id"] = call_id
        return result
    
    def _calculate_metrics(
        self,
        model_name: str,
        successful_results: List[Dict[str, Any]],
        failed_results: List[Dict[str, Any]],
        total_time: float,
        total_calls: int
    ) -> Dict[str, Any]:
        successful_calls = len(successful_results)
        failed_calls = len(failed_results)
        
        # Response time metrics
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        # Token metrics
        total_input_tokens = sum(r["input_tokens"] for r in successful_results)
        total_output_tokens = sum(r["output_tokens"] for r in successful_results)
        
        # Cost calculation
        total_cost = self._calculate_cost(model_name, total_input_tokens, total_output_tokens)
        cost_per_call = total_cost / successful_calls if successful_calls > 0 else 0
        
        # Error categorization
        error_counts = {"timeout": 0, "rate_limit": 0, "service_error": 0, "auth_error": 0, "validation_error": 0, "exception": 0}
        for result in failed_results:
            error_type = result.get("error", {}).get("type", "unknown")
            if error_type in error_counts:
                error_counts[error_type] += 1
            else:
                error_counts["service_error"] += 1
        
        # Calculate rates
        success_rate = (successful_calls / total_calls) * 100 if total_calls > 0 else 0
        error_rate = (failed_calls / total_calls) * 100 if total_calls > 0 else 0
        throughput = total_calls / total_time if total_time > 0 else 0
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "throughput": throughput,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost": total_cost,
            "cost_per_call": cost_per_call,
            "errors_timeout": error_counts["timeout"],
            "errors_rate_limit": error_counts["rate_limit"],
            "errors_service": error_counts["service_error"],
            "errors_auth": error_counts["auth_error"],
            "errors_validation": error_counts["validation_error"],
            "errors_exception": error_counts["exception"]
        }
    
    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.pricing_data.get(model_name)
        if not pricing:
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing["input_cost_per_1k_tokens"]
        output_cost = (output_tokens / 1000) * pricing["output_cost_per_1k_tokens"]
        
        return input_cost + output_cost
    
    async def run_single_prompt_benchmark(
        self,
        prompt: str,
        prompt_info: Dict[str, Any],
        number_of_calls: int = 100,
        specific_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        models_to_test = self.models_config["models"]
        
        if specific_model:
            models_to_test = [
                model for model in models_to_test 
                if model["name"] == specific_model
            ]
            if not models_to_test:
                raise ValueError(f"Model '{specific_model}' not found in configuration")
        
        results = []
        
        for model_config in models_to_test:
            try:
                result = await self.benchmark_model(
                    model_config,
                    prompt,
                    prompt_info,
                    number_of_calls
                )
                results.append(result)
                
                print(f"Completed benchmark for {model_config['name']}")
                print(f"  Success rate: {result['metrics']['success_rate']:.1f}%")
                print(f"  Average response time: {result['metrics']['avg_response_time']:.3f}s")
                print(f"  Total cost: ${result['metrics']['total_cost']:.6f}")
                print()
                
            except Exception as e:
                print(f"Failed to benchmark {model_config['name']}: {str(e)}")
                continue
        
        return results

    async def run_multi_prompt_benchmark(
        self,
        number_of_calls: int = 100,
        specific_model: Optional[str] = None,
        specific_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        prompts_to_test = self.prompts_config["prompts"]
        
        if specific_prompt:
            prompts_to_test = [
                prompt for prompt in prompts_to_test 
                if prompt["name"] == specific_prompt
            ]
            if not prompts_to_test:
                raise ValueError(f"Prompt '{specific_prompt}' not found in configuration")
        
        all_results = []
        prompt_results = {}
        
        for prompt_config in prompts_to_test:
            prompt_name = prompt_config["name"]
            prompt_text = prompt_config["prompt"]
            
            print(f"Running benchmark for prompt: {prompt_name}")
            print(f"Description: {prompt_config['description']}")
            print()
            
            results = await self.run_single_prompt_benchmark(
                prompt=prompt_text,
                prompt_info=prompt_config,
                number_of_calls=number_of_calls,
                specific_model=specific_model
            )
            
            prompt_results[prompt_name] = results
            all_results.extend(results)
            
            print(f"Completed all models for prompt: {prompt_name}")
            print("-" * 50)
        
        return {
            "all_results": all_results,
            "prompt_results": prompt_results
        }