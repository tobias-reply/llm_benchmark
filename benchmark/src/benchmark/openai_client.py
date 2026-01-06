import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError


class OpenAIClient:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Check if custom base URL is provided (e.g., for Azure OpenAI)
        base_url = os.getenv('OPENAI_BASE_URL')

        if base_url:
            # Ensure Azure OpenAI base URL ends with /openai/v1/
            if 'azure' in base_url.lower() and not base_url.endswith('/openai/v1/'):
                if not base_url.endswith('/'):
                    base_url += '/'
                base_url += 'openai/v1/'

            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)

    async def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        region: Optional[str] = None  # Not used for OpenAI, kept for API compatibility
    ) -> Dict[str, Any]:
        start_time = time.time()
        invocation_timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Build request parameters
            request_params = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            # Only include max_tokens if it's specified
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # Only include temperature if it's specified
            if temperature is not None:
                request_params["temperature"] = temperature

            # Call OpenAI API using chat completions
            response = await self.client.chat.completions.create(**request_params)

            response_time = time.time() - start_time
            response_timestamp = datetime.now(timezone.utc).isoformat()

            # Extract content from response
            content = ""
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content or ""

            # Extract token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

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

        except RateLimitError as e:
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
                    "type": "rate_limit",
                    "code": "RateLimitError",
                    "message": str(e)
                }
            }

        except APITimeoutError as e:
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
                    "type": "timeout",
                    "code": "APITimeoutError",
                    "message": str(e)
                }
            }

        except APIConnectionError as e:
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
                    "code": "APIConnectionError",
                    "message": str(e)
                }
            }

        except APIError as e:
            # Categorize API errors
            error_type = "service_error"
            if "authentication" in str(e).lower() or "authorization" in str(e).lower():
                error_type = "auth_error"
            elif "invalid" in str(e).lower() or "validation" in str(e).lower():
                error_type = "validation_error"

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
                    "type": error_type,
                    "code": "APIError",
                    "message": str(e)
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
