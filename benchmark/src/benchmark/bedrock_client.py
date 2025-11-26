import asyncio
import json
import time
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, BotoCoreError


class BedrockClient:
    def __init__(self, region_name: str = "eu-central-1"):
        self.region_name = region_name
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
    
    async def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Prepare request body based on model provider
            if "anthropic" in model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
            elif "amazon" in model_id:
                if "nova" in model_id:
                    # Amazon Nova models use the messages format
                    body = {
                        "messages": [{"role": "user", "content": [{"text": prompt}]}],
                        "inferenceConfig": {
                            "maxTokens": max_tokens,
                            "temperature": temperature,
                            "topP": 0.9
                        }
                    }
                else:
                    # Amazon Titan models use the old format
                    body = {
                        "inputText": prompt,
                        "textGenerationConfig": {
                            "maxTokenCount": max_tokens,
                            "temperature": temperature,
                            "topP": 0.9
                        }
                    }
            elif "meta" in model_id:
                body = {
                    "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
            elif "mistral" in model_id:
                body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
            elif "openai" in model_id:
                body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            elif "qwen" in model_id:
                body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
            elif "cohere" in model_id:
                body = {
                    "query": prompt,
                    "documents": [],
                    "top_k": 10,
                    "return_documents": False
                }
            else:
                # Default format for other providers
                body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            
            # Make the API call
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json"
                )
            )
            
            response_time = time.time() - start_time
            response_body = json.loads(response["body"].read())
            
            # Parse response based on model provider
            if "anthropic" in model_id:
                content = response_body.get("content", [{}])[0].get("text", "")
                input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
                output_tokens = response_body.get("usage", {}).get("output_tokens", 0)
            elif "amazon" in model_id:
                if "nova" in model_id:
                    # Amazon Nova models use the messages format
                    content = response_body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
                    usage = response_body.get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)
                else:
                    # Amazon Titan models use the old format
                    results = response_body.get("results", [{}])
                    content = results[0].get("outputText", "") if results else ""
                    input_tokens = response_body.get("inputTextTokenCount", 0)
                    output_tokens = results[0].get("tokenCount", 0) if results else 0
            elif "meta" in model_id:
                content = response_body.get("generation", "")
                # Clean up any trailing special tokens
                if content.endswith("<|eot_id|>"):
                    content = content[:-10]
                input_tokens = response_body.get("prompt_token_count", 0)
                output_tokens = response_body.get("generation_token_count", 0)
            elif "mistral" in model_id:
                choices = response_body.get("choices", [{}])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                usage = response_body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            elif "openai" in model_id:
                choices = response_body.get("choices", [{}])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                usage = response_body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            elif "qwen" in model_id:
                content = response_body.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = response_body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            elif "cohere" in model_id:
                # Cohere rerank returns relevance scores, not text generation
                results = response_body.get("results", [])
                content = f"Rerank results: {len(results)} documents processed"
                input_tokens = response_body.get("meta", {}).get("api_version", {}).get("billed_units", {}).get("input_tokens", 0)
                output_tokens = 0  # Rerank doesn't generate text
            else:
                # Default parsing
                content = response_body.get("completion", response_body.get("text", ""))
                input_tokens = response_body.get("input_tokens", len(prompt.split()) * 1.3)  # Rough estimate
                output_tokens = response_body.get("output_tokens", len(content.split()) * 1.3)
            
            return {
                "success": True,
                "response_time": response_time,
                "response": content,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "error": None
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": {
                    "type": self._categorize_error(error_code),
                    "code": error_code,
                    "message": error_message
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "response": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": {
                    "type": "service_error",
                    "code": "UnknownError",
                    "message": str(e)
                }
            }
    
    def _categorize_error(self, error_code: str) -> str:
        if "Throttling" in error_code or "TooManyRequests" in error_code:
            return "rate_limit"
        elif "Timeout" in error_code or "RequestTimeout" in error_code:
            return "timeout"
        elif "ValidationException" in error_code:
            return "validation_error"
        elif "AccessDenied" in error_code:
            return "auth_error"
        else:
            return "service_error"