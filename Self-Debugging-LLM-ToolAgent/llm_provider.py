"""Free and local LLM provider adapters.

llama.cpp is the default: it runs the bundled local GGUF model with no API key.
Ollama and Gemini remain optional alternatives.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    model: str
    available: bool
    detail: str


class OllamaProvider:
    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ) -> None:
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> ProviderStatus:
        try:
            request = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(request, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = [item.get("name", "") for item in payload.get("models", [])]
            exact_match = self.model in models
            prefix_match = any(name.startswith(self.model + ":") for name in models) if ":" not in self.model else False
            if exact_match or prefix_match:
                return ProviderStatus(self.provider, self.model, True, "Running")
            return ProviderStatus(
                self.provider,
                self.model,
                False,
                f"Model is not installed. Run: ollama pull {self.model}",
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            return ProviderStatus(
                self.provider, self.model, False,
                f"Ollama is not reachable at {self.base_url}: {exc}",
            )

    def generate(self, prompt: str, system: str = "") -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if "error" in result:
            raise RuntimeError(result["error"])
        return str(result.get("response", "")).strip()


class GeminiProvider:
    """Google Gemini API adapter for GEMINI_API_KEY free-tier use."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.provider = "gemini"
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.timeout = timeout

    def status(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus(
                self.provider, self.model, False,
                "Set GEMINI_API_KEY to use the free Gemini API tier.",
            )
        return ProviderStatus(self.provider, self.model, True, "API key configured")

    def generate(self, prompt: str, system: str = "") -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        parts = []
        if system:
            parts.append({"text": system + "\n\n"})
        parts.append({"text": prompt})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 800},
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        candidates = result.get("candidates", [])
        if not candidates:
            raise RuntimeError(result.get("error", {}).get("message", "Gemini returned no result"))
        content = candidates[0].get("content", {})
        return "".join(part.get("text", "") for part in content.get("parts", [])).strip()


class LlamaCppProvider:
    """No-account local provider using llama.cpp and a downloaded GGUF model."""

    DEFAULT_MODEL = "/Users/ramavathramesh/.cache/llm-models/SmolLM2-360M-Instruct-Q4_K_M.gguf"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        binary: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.provider = "llama_cpp"
        self.model = model
        self.binary = binary or shutil.which("llama-cli") or "/opt/homebrew/bin/llama-cli"
        self.timeout = timeout

    def status(self) -> ProviderStatus:
        if not os.path.isfile(self.model):
            return ProviderStatus(
                self.provider, self.model, False,
                "GGUF model file is not installed.",
            )
        if not os.path.isfile(self.binary):
            return ProviderStatus(
                self.provider, self.model, False,
                "llama-cli binary is not installed.",
            )
        return ProviderStatus(
            self.provider, self.model, True,
            "Local llama.cpp model is ready",
        )

    def generate(self, prompt: str, system: str = "") -> str:
        full_prompt = (
            f"{system}\n\nUser: {prompt}\nAssistant:"
            if system
            else f"User: {prompt}\nAssistant:"
        )
        completed = subprocess.run(
            [
                self.binary,
                "-m",
                self.model,
                "-p",
                full_prompt,
                "-n",
                "300",
                "--temp",
                "0",
                "--no-display-prompt",
                "--log-disable",
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "llama.cpp generation failed")
        output = completed.stdout.strip()
        marker = "Assistant:"
        if marker in output:
            output = output.split(marker, 1)[1].strip()
        for terminator in ("\n[ Prompt:", "\n[ Prompt ", "\n>", "\nExiting..."):
            if terminator in output:
                output = output.split(terminator, 1)[0].strip()
        return output or "The local model returned an empty response."


def create_provider(
    provider: str = "llama_cpp",
    model: str | None = None,
) -> OllamaProvider | GeminiProvider | LlamaCppProvider:
    provider = provider.lower().strip()
    if provider == "gemini":
        return GeminiProvider(model=model or "gemini-2.0-flash")
    if provider == "llama_cpp":
        return LlamaCppProvider(model=model or os.getenv("LLAMA_MODEL", LlamaCppProvider.DEFAULT_MODEL))
    return OllamaProvider(
        model=model or "llama3.2:1b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
