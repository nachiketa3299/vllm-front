import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional

import httpx

from .config import AppConfig
from .models import AppError, ProbedModelInfo, RequestLog


class VLLMClient:
    _PROBE_TTL_SECONDS = 5.0
    _PROBE_TIMEOUT_SECONDS = 5.0

    def __init__(self, config: AppConfig):
        self.config = config
        self._probe_cache: tuple[float, ProbedModelInfo] | None = None
        self._probe_lock = asyncio.Lock()

    async def probe_model_info(self, *, force: bool = False) -> Optional[ProbedModelInfo]:
        if not force:
            cached = self._cached_probe()
            if cached is not None:
                return cached

        async with self._probe_lock:
            if not force:
                cached = self._cached_probe()
                if cached is not None:
                    return cached

            info = await self._fetch_model_info()
            if info is not None:
                self._probe_cache = (time.monotonic(), info)
            else:
                self._probe_cache = None
            return info

    def _cached_probe(self) -> Optional[ProbedModelInfo]:
        if self._probe_cache is None:
            return None
        cached_at, info = self._probe_cache
        if time.monotonic() - cached_at > self._PROBE_TTL_SECONDS:
            return None
        return info

    async def _fetch_model_info(self) -> Optional[ProbedModelInfo]:
        endpoint = f"{self.config.vllm_base_url.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=self._PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None
        first = data[0] if isinstance(data[0], dict) else {}

        model_id = first.get("id")
        if not isinstance(model_id, str) or not model_id:
            return None

        raw_len = first.get("max_model_len")
        try:
            max_model_len = int(raw_len) if raw_len is not None else None
        except (TypeError, ValueError):
            max_model_len = None

        return ProbedModelInfo(
            model=model_id,
            max_model_len=max_model_len,
        )

    async def require_model_info(self) -> ProbedModelInfo:
        info = await self.probe_model_info()
        if info is None:
            raise AppError(
                503,
                f"vLLM 서버에 연결할 수 없습니다: {self.config.vllm_base_url}",
            )
        return info

    def create_payload(
        self,
        *,
        model: str,
        system_prompt: Optional[str],
        user_text: Optional[str],
        image_data_url: Optional[str],
        max_completion_tokens: Optional[int] = None,
        response_format: dict[str, Any] | None = None,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = []
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        if image_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        payload = {
            "model": model,
            "temperature": float(temperature),
            "max_completion_tokens": max_completion_tokens or self.config.max_completion_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
            "messages": messages,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    async def create_chat_completion(
        self,
        *,
        system_prompt: Optional[str],
        user_text: Optional[str],
        image_data_url: Optional[str],
        max_completion_tokens: Optional[int],
        timeout_seconds: Optional[int],
        response_format: dict[str, Any] | None,
        enable_thinking: bool,
        temperature: float,
        log: RequestLog,
        label: str,
    ) -> dict[str, Any]:
        info = await self.require_model_info()
        log.add(
            f"Sending {label} request to vLLM at {self.config.vllm_base_url} (model={info.model})"
        )

        try:
            async with httpx.AsyncClient(
                timeout=timeout_seconds or self.config.timeout_seconds
            ) as client:
                response = await client.post(
                    f"{self.config.vllm_base_url.rstrip('/')}/chat/completions",
                    json=self.create_payload(
                        model=info.model,
                        system_prompt=system_prompt,
                        user_text=user_text,
                        image_data_url=image_data_url,
                        max_completion_tokens=max_completion_tokens,
                        response_format=response_format,
                        enable_thinking=enable_thinking,
                        temperature=temperature,
                    ),
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            if "cannot identify image file" in body:
                raise AppError(
                    502,
                    (
                        "현재 실행 중인 vLLM 인스턴스가 이미지 입력을 디코드하지 못하고 있습니다. "
                        "앱 업로드 자체 문제라기보다 vLLM 설정 또는 버전 문제일 가능성이 큽니다. "
                        f"raw={body}"
                    ),
                ) from exc
            raise AppError(
                502, f"vLLM returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.RequestError as exc:
            raise AppError(
                502, f"Failed to connect to vLLM at {self.config.vllm_base_url}."
            ) from exc
        except ValueError as exc:
            raise AppError(502, "vLLM returned an invalid JSON response.") from exc

        log.add(f"Received {label} response from vLLM")
        return payload

    async def stream_chat_completion(
        self,
        *,
        system_prompt: Optional[str],
        user_text: Optional[str],
        image_data_url: Optional[str],
        max_completion_tokens: Optional[int],
        timeout_seconds: Optional[int],
        response_format: dict[str, Any] | None,
        enable_thinking: bool,
        temperature: float,
        log: RequestLog,
    ) -> AsyncIterator[dict[str, str]]:
        """vLLM /chat/completions 를 stream=true 로 호출. 한 chunk 당
        ``{"reasoning": ..., "content": ...}`` 둘 중 하나(또는 둘 다)를 yield."""
        info = await self.require_model_info()
        log.add(
            f"Streaming from vLLM at {self.config.vllm_base_url} "
            f"(model={info.model})"
        )

        payload = self.create_payload(
            model=info.model,
            system_prompt=system_prompt,
            user_text=user_text,
            image_data_url=image_data_url,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )
        payload["stream"] = True

        url = f"{self.config.vllm_base_url.rstrip('/')}/chat/completions"
        timeout = timeout_seconds or self.config.timeout_seconds

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        raise AppError(
                            502,
                            f"vLLM returned HTTP {response.status_code}: {body}",
                        )
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        chunk: dict[str, str] = {}
                        # vLLM streaming: 키 이름이 'reasoning' (non-streaming 은
                        # 'reasoning_content'). 양쪽 모두 방어적으로 받음.
                        reasoning = (
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                        )
                        if isinstance(reasoning, str) and reasoning:
                            chunk["reasoning"] = reasoning
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            chunk["content"] = content
                        if chunk:
                            yield chunk
        except httpx.RequestError as exc:
            raise AppError(
                502, f"Failed to connect to vLLM at {self.config.vllm_base_url}."
            ) from exc
