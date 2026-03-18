import json
import os
import asyncio
import time
import websockets
import opuslib_next
from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    """Deepgram Streaming ASR Provider

    Uses Deepgram Nova-2/Nova-3 WebSocket API for real-time speech recognition.
    Sends raw binary PCM audio frames — no base64 encoding overhead.
    Supports multilingual detection including Hindi+English code-switching.
    """

    def __init__(self, config, delete_audio_file):
        super().__init__()
        self.interface_type = InterfaceType.STREAM
        self.config = config
        self.text = ""
        self.decoder = opuslib_next.Decoder(16000, 1)
        self.asr_ws = None
        self.forward_task = None
        self.is_processing = False
        self.server_ready = False

        # Deepgram configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Deepgram API key is required")

        # Model and language configuration
        self.model = config.get("model", "nova-3")
        # "multi" enables automatic multilingual detection (Hindi+English etc.)
        self.language = config.get("language", "multi")

        # Audio configuration
        self.sample_rate = 16000
        self.encoding = "linear16"
        self.channels = 1

        # Recognition settings
        self.interim_results = config.get("interim_results", True)
        self.punctuate = config.get("punctuate", True)
        self.smart_format = config.get("smart_format", True)
        # Endpointing: ms of silence to detect end of speech
        self.endpointing = config.get("endpointing", 300)

        # Build WebSocket URL with query parameters
        params = (
            f"encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
            f"&channels={self.channels}"
            f"&model={self.model}"
            f"&language={self.language}"
            f"&interim_results={'true' if self.interim_results else 'false'}"
            f"&punctuate={'true' if self.punctuate else 'false'}"
            f"&smart_format={'true' if self.smart_format else 'false'}"
            f"&endpointing={self.endpointing}"
        )
        self.ws_url = f"wss://api.deepgram.com/v1/listen?{params}"

        self.output_dir = config.get("output_dir", "tmp/")
        self.delete_audio_file = delete_audio_file
        os.makedirs(self.output_dir, exist_ok=True)

        logger.bind(tag=TAG).info(
            f"Deepgram ASR initialized | Model: {self.model} | Language: {self.language}"
        )

    async def open_audio_channels(self, conn):
        await super().open_audio_channels(conn)

    async def receive_audio(self, conn, audio, audio_have_voice):
        """Receive and process audio data — stream to Deepgram in real-time"""
        # Initialize voiceprint audio cache
        if not hasattr(conn, "asr_audio_for_voiceprint"):
            conn.asr_audio_for_voiceprint = []

        # Store audio for voiceprint
        if audio:
            conn.asr_audio_for_voiceprint.append(audio)

        conn.asr_audio.append(audio)
        conn.asr_audio = conn.asr_audio[-10:]

        # Start WebSocket connection when voice is first detected
        if audio_have_voice and not self.is_processing and not self.asr_ws:
            try:
                await self._start_recognition(conn)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to start recognition: {e}")
                await self._cleanup()
                return

        # Send raw binary PCM audio to Deepgram (no base64, no JSON wrapping)
        if self.asr_ws and self.is_processing and self.server_ready:
            try:
                pcm_frame = self.decoder.decode(audio, 960)
                # Deepgram accepts raw binary audio — much faster than base64
                await self.asr_ws.send(pcm_frame)
            except Exception as e:
                logger.bind(tag=TAG).warning(f"Failed to send audio: {e}")
                await self._cleanup(conn)

    async def _start_recognition(self, conn):
        """Establish WebSocket connection to Deepgram"""
        try:
            self.asr_ws = await websockets.connect(
                self.ws_url,
                additional_headers={"Authorization": f"Token {self.api_key}"},
                max_size=1000000000,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )

            logger.bind(tag=TAG).info("Deepgram WebSocket connected")
            self.is_processing = True
            self.server_ready = True
            self._final_segments = []  # Buffer for is_final segments
            self._recognition_start = time.time()

            # Start listening for results
            self.forward_task = asyncio.create_task(self._forward_results(conn))

            # Send cached audio (buffered before connection was established)
            if conn.asr_audio:
                for cached_audio in conn.asr_audio[-10:]:
                    try:
                        pcm_frame = self.decoder.decode(cached_audio, 960)
                        await self.asr_ws.send(pcm_frame)
                    except Exception as e:
                        logger.bind(tag=TAG).warning(f"Failed to send cached audio: {e}")
                        break

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to connect to Deepgram: {e}")
            await self._cleanup()
            raise

    async def _forward_results(self, conn):
        """Listen for transcription results from Deepgram"""
        try:
            while not conn.stop_event.is_set():
                try:
                    response = await asyncio.wait_for(self.asr_ws.recv(), timeout=1.0)
                    result = json.loads(response)

                    msg_type = result.get("type", "")

                    if msg_type == "Results":
                        is_final = result.get("is_final", False)
                        speech_final = result.get("speech_final", False)

                        # Extract transcript
                        channel = result.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        transcript = ""
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")

                        if not transcript:
                            continue

                        if is_final:
                            # Accumulate finalized segments
                            self._final_segments.append(transcript)
                            logger.bind(tag=TAG).debug(f"Final segment: {transcript}")

                        if speech_final:
                            # End of utterance — combine all final segments
                            full_text = " ".join(self._final_segments).strip()
                            if not full_text:
                                continue

                            elapsed = time.time() - self._recognition_start
                            logger.bind(tag=TAG).info(
                                f"ASR completed in {elapsed:.3f}s | text: {full_text}"
                            )

                            # Process the result
                            if conn.client_listen_mode == "manual":
                                if self.text:
                                    self.text += " " + full_text
                                else:
                                    self.text = full_text

                                if conn.client_voice_stop:
                                    audio_data = getattr(conn, "asr_audio_for_voiceprint", [])
                                    if len(audio_data) > 0:
                                        await self.handle_voice_stop(conn, audio_data)
                                        conn.asr_audio.clear()
                                        conn.reset_audio_states()
                                    break
                            else:
                                # Automatic/realtime mode — process immediately
                                self.text = full_text
                                conn.reset_audio_states()
                                audio_data = getattr(conn, "asr_audio_for_voiceprint", [])
                                await self.handle_voice_stop(conn, audio_data)
                                break

                    elif msg_type == "Metadata":
                        logger.bind(tag=TAG).debug(f"Deepgram metadata: {result}")

                    elif msg_type == "SpeechStarted":
                        logger.bind(tag=TAG).debug("Deepgram: speech started")

                    elif msg_type == "UtteranceEnd":
                        # Utterance boundary — process accumulated segments
                        if self._final_segments:
                            full_text = " ".join(self._final_segments).strip()
                            if full_text:
                                elapsed = time.time() - self._recognition_start
                                logger.bind(tag=TAG).info(
                                    f"ASR completed (utterance end) in {elapsed:.3f}s | text: {full_text}"
                                )

                                self.text = full_text
                                conn.reset_audio_states()
                                audio_data = getattr(conn, "asr_audio_for_voiceprint", [])
                                await self.handle_voice_stop(conn, audio_data)
                                break

                    elif msg_type == "Error":
                        error_msg = result.get("message", "Unknown error")
                        logger.bind(tag=TAG).error(f"Deepgram error: {error_msg}")
                        break

                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    logger.bind(tag=TAG).info("Deepgram connection closed")
                    self.is_processing = False
                    break
                except json.JSONDecodeError as e:
                    logger.bind(tag=TAG).warning(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Error processing result: {e}")
                    break

        except Exception as e:
            logger.bind(tag=TAG).error(f"Result forwarding failed: {e}")
        finally:
            await self._cleanup()
            if conn:
                if hasattr(conn, "asr_audio_for_voiceprint"):
                    conn.asr_audio_for_voiceprint = []
                if hasattr(conn, "asr_audio"):
                    conn.asr_audio = []

    async def _cleanup(self, conn=None):
        """Cleanup resources"""
        self.is_processing = False
        self.server_ready = False
        self._final_segments = []

        if self.asr_ws:
            try:
                # Send CloseStream for graceful shutdown
                await self.asr_ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.wait_for(self.asr_ws.close(), timeout=2.0)
            except Exception:
                pass
            finally:
                self.asr_ws = None

        self.forward_task = None

    async def speech_to_text(self, opus_data, session_id, audio_format="opus", artifacts=None):
        """Get recognition result (called by base class for streaming providers)"""
        result = self.text
        self.text = ""
        return result, None

    async def close(self):
        """Close resources"""
        await self._cleanup(None)
        if hasattr(self, "decoder") and self.decoder is not None:
            try:
                del self.decoder
                self.decoder = None
            except Exception as e:
                logger.bind(tag=TAG).debug(f"Error releasing decoder: {e}")
