import io
import time
import wave
import os
import json
import aiohttp
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        self.interface_type = InterfaceType.NON_STREAM
        self.api_key = config.get("api_key")
        self.api_url = config.get("base_url", "https://api.openai.com/v1/audio/transcriptions")
        self.model = config.get("model_name", "gpt-4o-mini-transcribe")
        self.language = config.get("language")
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file

        os.makedirs(self.output_dir, exist_ok=True)

    def requires_file(self) -> bool:
        # No longer need disk file — we send WAV from memory
        return False

    async def speech_to_text(self, opus_data: List[bytes], session_id: str, audio_format="opus", artifacts=None) -> Tuple[Optional[str], Optional[str]]:
        try:
            if artifacts is None:
                return "", None

            # Build WAV in memory from PCM bytes — skip disk I/O
            pcm_bytes = artifacts.pcm_bytes
            if not pcm_bytes:
                return "", None

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_bytes)
            wav_buffer.seek(0)

            start_time = time.time()

            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }

            # Build multipart form data
            form = aiohttp.FormData()
            form.add_field("file", wav_buffer, filename="audio.wav", content_type="audio/wav")
            form.add_field("model", self.model)

            # Add language hint if configured
            if self.language:
                form.add_field("language", self.language)

            # Use streaming SSE for faster response with transcribe models
            use_stream = "transcribe" in self.model
            if use_stream:
                form.add_field("stream", "true")

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, data=form) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error: {response.status} - {error_text}")

                    if use_stream:
                        # Parse SSE stream for transcript.text.done event
                        text = await self._parse_sse_stream(response)
                    else:
                        # Non-streaming: parse JSON response
                        result = await response.json()
                        text = result.get("text", "")

            elapsed = time.time() - start_time
            logger.bind(tag=TAG).info(f"ASR completed in {elapsed:.3f}s | text: {text}")

            return text, None

        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}")
            return "", None

    async def _parse_sse_stream(self, response: aiohttp.ClientResponse) -> str:
        """Parse SSE stream from OpenAI streaming transcription API.
        Returns the final transcribed text."""
        full_text = ""

        async for line in response.content:
            line = line.decode("utf-8").strip()

            if not line:
                continue

            # SSE format: "data: {...}" or "event: ..."
            if line.startswith("data: "):
                data_str = line[6:]  # strip "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    event_type = data.get("type", "")

                    if event_type == "transcript.text.done":
                        # Final result — use this
                        full_text = data.get("text", "")
                        break
                    elif event_type == "transcript.text.delta":
                        # Accumulate deltas as fallback
                        full_text += data.get("delta", "")

                except json.JSONDecodeError:
                    continue

        return full_text
