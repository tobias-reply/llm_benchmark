import asyncio
import json
import time
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, BotoCoreError


class BedrockClient:
    def __init__(self, region_name: str = "eu-central-1"):
        self.default_region = region_name
        self.region_name = region_name  # Keep for backward compatibility
        self.clients = {}  # Cache for region-specific clients

        # Check authentication method
        bearer_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        if bearer_token:
            print("✅ Using AWS Bearer Token authentication from environment")
        else:
            print("✅ Using AWS Profile/IAM authentication")

        self.client = self._get_client_for_region(region_name)

    def _get_client_for_region(self, region_name: str):
        """Get or create a cached client for the specified region.

        Boto3 will automatically use AWS_BEARER_TOKEN_BEDROCK if set,
        otherwise it falls back to the credential chain (profile, IAM role, etc.)
        """
        if region_name not in self.clients:
            self.clients[region_name] = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name
            )
        return self.clients[region_name]
    
    async def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        invocation_timestamp = datetime.now(timezone.utc).isoformat()

        # Use specified region or default
        target_region = region or self.default_region
        client = self._get_client_for_region(target_region)

        try:
            # Prepare inference config based on model provider
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
            
            # Claude models don't support both temperature and topP
            if "anthropic" not in model_id:
                inference_config["topP"] = 0.9
            
            # Use Converse API for all models - unified interface with consistent token usage
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.converse(
                    modelId=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    inferenceConfig=inference_config
                )
            )

            response_time = time.time() - start_time
            response_timestamp = datetime.now(timezone.utc).isoformat()

            # Extract content from unified Converse API response
            content = ""
            if "output" in response:
                output = response["output"]
                if "message" in output:
                    message = output["message"]
                    if "content" in message:
                        content_blocks = message["content"]
                        for block in content_blocks:
                            if "text" in block:
                                content += block["text"]
            
            # Extract token usage from unified Converse API response
            usage_info = response.get("usage", {})
            input_tokens = usage_info.get("inputTokens", 0)
            output_tokens = usage_info.get("outputTokens", 0)
            
            return {
                "success": True,
                "response_time": response_time,
                "invocation_timestamp": invocation_timestamp,
                "response_timestamp": response_timestamp,
                "response": content,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "error": None
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            error_timestamp = datetime.now(timezone.utc).isoformat()

            return {
                "success": False,
                "response_time": time.time() - start_time,
                "invocation_timestamp": invocation_timestamp,
                "response_timestamp": error_timestamp,
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
            error_timestamp = datetime.now(timezone.utc).isoformat()
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "invocation_timestamp": invocation_timestamp,
                "response_timestamp": error_timestamp,
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