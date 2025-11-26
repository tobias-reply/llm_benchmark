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
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": 0.9
                    }
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
                results = response_body.get("results", [{}])
                content = results[0].get("outputText", "") if results else ""
                input_tokens = response_body.get("inputTextTokenCount", 0)
                output_tokens = results[0].get("tokenCount", 0) if results else 0
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