"""Hume.ai EVI (Empathic Voice Interface) Provider

This provider integrates Hume.ai's EVI API which combines ASR, LLM, and TTS
with emotional intelligence in a single low-latency WebSocket connection.

Architecture:
    Audio Input → Hume EVI (ASR+LLM+Emotion) → Function Call Detection
                                           ↓
                                  Execute via UnifiedToolHandler
                                           ↓
                                  Return to Hume EVI
                                           ↓
                                  TTS Output → Audio

Key Features:
- Emotionally intelligent responses with prosody analysis
- Always interruptible conversation
- Multi-language support (11 languages)
- Vocal expression measurement
- Custom voice and personality configuration
"""

import json
import uuid
import base64
import asyncio
import websockets
import opuslib_next
import numpy as np
from scipy import signal
from typing import Dict, Any, Optional
from config.logger import setup_logging
from core.utils.dialogue import Message
from core.handle.reportHandle import enqueue_asr_report, enqueue_tts_report
from core.utils import textUtils
from core.providers.realtime.hume_config_manager import auto_create_config_if_needed

TAG = __name__
logger = setup_logging()


class HumeRealtimeProvider:
    """Hume.ai EVI Provider

    Combines ASR, LLM, TTS with emotional intelligence in a single WebSocket connection.
    Integrates with existing tool execution infrastructure.
    """

    def __init__(self, config: Dict[str, Any], conn):
        """Initialize Hume EVI provider

        Args:
            config: Provider configuration containing:
                - api_key: Hume API key
                - config_id: Optional EVI configuration ID
                - language: Language code (default: en)
                - instructions: System instructions/prompt
                - temperature: Model temperature (default: 0.8)
            conn: Connection handler instance
        """
        self.conn = conn
        self.config = config

        # Hume API configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Hume API key is required for EVI provider")

        # Configuration
        self.config_id = config.get("config_id", "")  # Optional EVI config ID
        self.language = config.get("language", "en")

        # Ensure temperature is float
        temp = config.get("temperature", 0.8)
        self.temperature = float(temp) if temp else 0.8

        # Audio configuration
        # Input: ESP32 sends 16kHz audio → send directly to Hume at 16kHz
        # Output: Will be determined by Hume's response (likely 24kHz or 48kHz)
        self.input_sample_rate = 16000   # 16kHz from ESP32
        self.hume_input_sample_rate = 16000  # Send 16kHz to Hume (no resampling)
        self.output_sample_rate = 24000  # Target output for ESP32 (24kHz hardware playback)
        self.channels = 1  # Mono
        self.audio_format = "linear16"  # 16-bit PCM
        self.frame_duration_ms = 60  # 60ms frames

        # Frame sizes for different sample rates
        self.input_frame_size = int(self.input_sample_rate * self.frame_duration_ms / 1000)   # 960 samples @ 16kHz
        self.hume_input_frame_size = int(self.hume_input_sample_rate * self.frame_duration_ms / 1000)  # 960 samples @ 16kHz
        self.output_frame_size = int(self.output_sample_rate * self.frame_duration_ms / 1000)  # 1440 samples @ 24kHz

        # Opus decoder for client audio at 16kHz (receiving from ESP32)
        self.opus_decoder = opuslib_next.Decoder(16000, 1)

        # Opus encoder for sending audio to client at 24kHz (sending to ESP32)
        self.opus_encoder = opuslib_next.Encoder(24000, 1, opuslib_next.APPLICATION_VOIP)

        # WebSocket connection
        self.ws = None
        self.ws_url = self._build_websocket_url()

        # Session state
        self.session_id = None
        self.is_connected = False
        self.is_processing = False
        self.is_music_playing = False  # Flag to pause audio processing during music playback

        # Tasks
        self.receive_task = None

        # Audio buffers
        self.audio_queue = asyncio.Queue()
        self.output_audio_buffer = []
        self.pcm_buffer = bytearray()  # Buffer for accumulating PCM16 samples

        # Conversation state
        self.response_in_progress = False
        self.audio_frames_sent = 0  # Track audio frames sent to device
        self.audio_frames_received = 0  # Track audio frames received from client

        # Keepalive tracking
        import time
        self.last_activity_time = time.time()

        # Interruption handling
        self.last_speech_started_time = 0
        self.response_start_time = 0

        # Tool/function call tracking
        # Note: Hume EVI doesn't have native function calling yet
        # We'll implement it via system prompt instructions
        self.pending_tool_calls = {}

        logger.bind(tag=TAG).info(
            f"Hume EVI provider initialized | "
            f"Language: {self.language} | Config ID: {self.config_id or 'default'}"
        )

    def _build_websocket_url(self) -> str:
        """Build Hume EVI WebSocket URL with authentication"""
        base_url = "wss://api.hume.ai/v0/evi/chat"
        params = [f"api_key={self.api_key}"]

        if self.config_id:
            params.append(f"config_id={self.config_id}")

        return f"{base_url}?{'&'.join(params)}"

    async def connect(self):
        """Establish WebSocket connection to Hume EVI"""
        try:
            # Auto-create or update Hume config with current tools if needed
            if not self.config_id and hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                logger.bind(tag=TAG).info("No config_id specified - attempting to auto-create Hume config with tools")

                # Get current tool definitions
                tools = self.conn.func_handler.get_functions()

                # Get system prompt and add memory if available
                system_prompt = self.conn.prompt if hasattr(self.conn, 'prompt') else None

                # Add memory context to system prompt
                if hasattr(self.conn, 'memory') and self.conn.memory:
                    try:
                        memory_context = await self.conn.memory.query_memory("")
                        if memory_context:
                            memory_section = (
                                "\n\n## MEMORY - WHAT YOU REMEMBER ABOUT THIS USER:\n"
                                f"{memory_context}\n"
                                "USE THIS INFORMATION TO PERSONALIZE YOUR RESPONSES.\n"
                            )
                            system_prompt = (system_prompt or "") + memory_section
                            logger.bind(tag=TAG).info("Added memory context to Hume config system prompt")
                    except Exception as e:
                        logger.bind(tag=TAG).warning(f"Failed to add memory context: {e}")

                # Auto-create config
                auto_config_id = await auto_create_config_if_needed(
                    api_key=self.api_key,
                    config_id=self.config_id,
                    tools=tools,
                    system_prompt=system_prompt,
                    config_name=f"xiaozhi-{self.conn.device_id}" if hasattr(self.conn, 'device_id') else "xiaozhi-auto"
                )

                if auto_config_id:
                    self.config_id = auto_config_id
                    # Rebuild WebSocket URL with new config_id
                    self.ws_url = self._build_websocket_url()
                    logger.bind(tag=TAG).success(f"Auto-created Hume config: {auto_config_id}")
                else:
                    logger.bind(tag=TAG).warning("Failed to auto-create config, will use default Hume config")

            logger.bind(tag=TAG).info(f"Connecting to Hume EVI: {self.ws_url.split('?')[0]}")

            # Connect to Hume EVI
            self.ws = await websockets.connect(
                self.ws_url,
                max_size=10000000,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            )

            self.session_id = str(uuid.uuid4())
            self.is_connected = True

            logger.bind(tag=TAG).info(f"Connected to Hume EVI | Session: {self.session_id}")

            # Configure session with audio settings
            await self._configure_session()

            # Start receive task
            self.receive_task = asyncio.create_task(self._receive_loop())

            return True

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to connect to Hume EVI: {e}")
            self.is_connected = False
            return False

    async def _configure_session(self):
        """Configure the Hume EVI session with audio settings and instructions"""
        try:
            # Send session settings for audio INPUT format (what we send TO Hume)
            # According to Hume docs, session_settings.audio configures the INPUT format
            session_settings = {
                "type": "session_settings",
                "audio": {
                    "encoding": self.audio_format,
                    "sample_rate": self.hume_input_sample_rate,  # Tell Hume we're sending 16kHz input
                    "channels": self.channels
                }
            }

            await self.ws.send(json.dumps(session_settings))
            logger.bind(tag=TAG).info(f"Session configured | Sending to Hume: {self.hume_input_sample_rate}Hz, Target ESP32 output: {self.output_sample_rate}Hz")

            # Clear audio buffers to ensure clean session start
            self.pcm_buffer = bytearray()
            self.audio_frames_sent = 0
            self.audio_frames_received = 0
            logger.bind(tag=TAG).info("Audio buffers cleared for new session")

            # Note: Hume EVI configuration (voice, personality, instructions) is set via config_id
            # or through the Hume web console. System instructions are part of the config.
            # Unlike OpenAI, we don't send instructions in the WebSocket protocol.

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to configure session: {e}")

    def _boost_audio_volume(self, pcm_data: bytes, gain_db: float = 6.0, ceiling: float = 0.89) -> bytes:
        """Boost audio volume with limiting to prevent clipping

        This is critical for Hume audio which is often too quiet.
        Based on the TypeScript implementation's boostLimitPCM16LEInPlace function.

        Args:
            pcm_data: PCM16 audio data (16-bit signed integers)
            gain_db: Gain in decibels (6dB = 2x amplitude)
            ceiling: Maximum amplitude as fraction of full scale (0.89 = 89% of max)

        Returns:
            Boosted PCM16 audio data
        """
        try:
            # Convert bytes to int16 numpy array
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)

            # Convert to float for processing (-1.0 to 1.0 range)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Apply gain (dB to linear: gain = 10^(dB/20))
            linear_gain = 10.0 ** (gain_db / 20.0)
            audio_float *= linear_gain

            # Apply ceiling limiter (hard limiting)
            max_amplitude = ceiling
            audio_float = np.clip(audio_float, -max_amplitude, max_amplitude)

            # Convert back to int16
            audio_int16_boosted = (audio_float * 32768.0).astype(np.int16)

            return audio_int16_boosted.tobytes()

        except Exception as e:
            logger.bind(tag=TAG).error(f"Audio boost error: {e}")
            return pcm_data  # Return original on error

    def _resample_audio(self, pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample PCM audio from one sample rate to another

        Args:
            pcm_data: PCM16 audio data
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled PCM16 audio data
        """
        try:
            # Convert bytes to int16 numpy array
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)

            # Calculate number of output samples
            num_samples = int(len(audio_int16) * to_rate / from_rate)

            # Resample using scipy
            resampled = signal.resample(audio_int16, num_samples)

            # Convert back to int16 and bytes
            resampled_int16 = np.clip(resampled, -32768, 32767).astype(np.int16)
            return resampled_int16.tobytes()

        except Exception as e:
            logger.bind(tag=TAG).error(f"Resampling error: {e}")
            return pcm_data  # Return original on error

    async def _receive_loop(self):
        """Main loop for receiving events from Hume EVI"""
        import time
        try:
            while self.is_connected and self.ws:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    event = json.loads(message)
                    await self._handle_event(event)

                except asyncio.TimeoutError:
                    # Check for idle timeout
                    idle_time = time.time() - self.last_activity_time

                    MAX_IDLE_TIME = 300  # 5 minutes
                    if idle_time > MAX_IDLE_TIME:
                        logger.bind(tag=TAG).warning(
                            f"Hume EVI connection idle for {idle_time:.1f}s - closing"
                        )
                        break

                    # Send ping
                    try:
                        await self.ws.ping()
                        logger.bind(tag=TAG).debug("Sent keepalive ping to Hume EVI")
                    except Exception as e:
                        logger.bind(tag=TAG).warning(f"Failed to send ping: {e}")
                        break
                    continue

                except websockets.exceptions.ConnectionClosed as e:
                    logger.bind(tag=TAG).warning(f"WebSocket connection closed: {e.code} - {e.reason}")
                    break

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in receive loop: {e}")
        finally:
            self.is_connected = False

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle events from Hume EVI

        Key event types:
        - audio_output: Streaming audio response from EVI
        - user_message: User's transcribed speech
        - assistant_message: EVI's text response
        - user_interruption: User interrupted the assistant
        - assistant_end: Assistant finished speaking
        - error: Error occurred
        """
        # Update activity timestamp
        import time
        self.last_activity_time = time.time()
        self.conn.last_activity_time = self.last_activity_time * 1000

        # Check for client abort
        if hasattr(self.conn, 'client_abort') and self.conn.client_abort:
            logger.bind(tag=TAG).info("Client abort detected - stopping response")
            self.conn.client_abort = False
            return

        event_type = event.get("type")

        # Audio output from EVI
        if event_type == "audio_output":
            await self._handle_audio_output(event)

        # User's transcribed message
        elif event_type == "user_message":
            await self._handle_user_message(event)

        # Assistant's text response
        elif event_type == "assistant_message":
            await self._handle_assistant_message(event)

        # User interruption
        elif event_type == "user_interruption":
            await self._handle_user_interruption(event)

        # Assistant finished speaking
        elif event_type == "assistant_end":
            await self._handle_assistant_end(event)

        # Tool/Function call
        elif event_type == "tool_call":
            await self._handle_tool_call(event)

        # Error
        elif event_type == "error":
            await self._handle_error(event)

        else:
            logger.bind(tag=TAG).debug(f"Unhandled event type: {event_type}")

    async def _handle_audio_output(self, event: Dict[str, Any]):
        """Handle audio output from Hume EVI"""
        try:
            # Mark response as in progress and send TTS signals
            if not self.response_in_progress:
                self.response_in_progress = True
                self.response_start_time = 0

                # Send BOTH initial start (enables codec) and sentence_start signals
                # This is critical - ESP32 needs the initial "start" signal to enable audio playback
                await self._send_tts_initial_start()
                await self._send_tts_start()

                # Reset audio sequence and frame counter
                self.conn._audio_sequence = 0
                self.audio_frames_sent = 0

                logger.bind(tag=TAG).info("Assistant response started - sent TTS start signals")

            # Get base64 encoded audio data
            audio_data_base64 = event.get("data", "")
            if not audio_data_base64:
                logger.bind(tag=TAG).warning("Received audio_output event with no data")
                return

            # Decode from base64 to get WAV file
            audio_wav = base64.b64decode(audio_data_base64)

            # Parse WAV header to extract actual sample rate and format
            if len(audio_wav) < 44:
                logger.bind(tag=TAG).error(f"WAV data too short: {len(audio_wav)} bytes")
                return

            # Parse WAV header (see http://soundfile.sapp.org/doc/WaveFormat/)
            # Bytes 22-23: Number of channels (2 bytes, little-endian)
            # Bytes 24-27: Sample rate (4 bytes, little-endian)
            # Bytes 34-35: Bits per sample (2 bytes, little-endian)
            wav_sample_rate = int.from_bytes(audio_wav[24:28], byteorder='little')
            wav_channels = int.from_bytes(audio_wav[22:24], byteorder='little')
            wav_bits_per_sample = int.from_bytes(audio_wav[34:36], byteorder='little')

            # Log WAV format info on first chunk
            if not hasattr(self, '_logged_wav_format'):
                logger.bind(tag=TAG).info(
                    f"Hume audio_output WAV format: {wav_sample_rate}Hz, "
                    f"{wav_channels} channel(s), {wav_bits_per_sample}-bit"
                )
                self._logged_wav_format = True

            logger.bind(tag=TAG).debug(
                f"Received audio chunk: {len(audio_wav)} bytes WAV "
                f"({wav_sample_rate}Hz, {wav_channels}ch, {wav_bits_per_sample}bit)"
            )

            # Extract PCM data from WAV (skip header)
            # WAV header size can vary, but standard PCM is 44 bytes
            # RIFF header (12 bytes) + fmt chunk (24 bytes) + data chunk header (8 bytes) = 44 bytes
            header_size = 44

            # Verify we have "data" chunk marker at expected position
            if len(audio_wav) >= 44:
                # Check for "data" marker at byte 36 (standard position)
                data_marker = audio_wav[36:40]
                if data_marker != b'data':
                    # Try to find "data" marker if not at standard position
                    data_pos = audio_wav.find(b'data')
                    if data_pos != -1:
                        header_size = data_pos + 8  # "data" + 4 bytes for data size
                        logger.bind(tag=TAG).debug(f"Non-standard WAV header size: {header_size} bytes")

            pcm_data = audio_wav[header_size:]

            # If Hume sends audio at a different sample rate than expected (24kHz),
            # we need to resample it
            if wav_sample_rate != self.output_sample_rate:
                logger.bind(tag=TAG).warning(
                    f"Resampling Hume output from {wav_sample_rate}Hz to {self.output_sample_rate}Hz"
                )
                pcm_data = self._resample_audio(pcm_data, wav_sample_rate, self.output_sample_rate)

            # Apply volume reduction to prevent echo/feedback with AEC
            # Hume's audio is often too loud and causes false interruptions with hardware AEC
            # Reduce by 6dB (half amplitude) to prevent overwhelming the echo canceller
            if not hasattr(self, '_logged_audio_reduction'):
                logger.bind(tag=TAG).info("Applying -6dB volume reduction to prevent AEC feedback")
                self._logged_audio_reduction = True

            # Analyze audio levels before reduction (first chunk only)
            if not hasattr(self, '_logged_audio_levels'):
                pcm_int16 = np.frombuffer(pcm_data, dtype=np.int16)
                rms = np.sqrt(np.mean(pcm_int16.astype(np.float32)**2))
                peak = np.max(np.abs(pcm_int16))
                logger.bind(tag=TAG).info(f"Hume audio levels BEFORE reduction: RMS={rms:.1f}, Peak={peak}/32768 ({peak/32768*100:.1f}%)")
                self._logged_audio_levels = True

            pcm_data = self._boost_audio_volume(pcm_data, gain_db=-6.0, ceiling=1.0)

            # Add to buffer
            self.pcm_buffer.extend(pcm_data)

            # Send complete frames to client at 24kHz (1440 samples = 60ms)
            FRAME_SIZE_BYTES = self.output_frame_size * 2  # 16-bit = 2 bytes per sample

            while len(self.pcm_buffer) >= FRAME_SIZE_BYTES:
                # Extract one frame
                frame_bytes = bytes(self.pcm_buffer[:FRAME_SIZE_BYTES])
                self.pcm_buffer = self.pcm_buffer[FRAME_SIZE_BYTES:]

                # Log first frame details
                if self.audio_frames_sent == 0:
                    logger.bind(tag=TAG).info(
                        f"Encoding first frame: {len(frame_bytes)} bytes PCM → Opus "
                        f"(frame_size={self.output_frame_size} samples @ {self.output_sample_rate}Hz)"
                    )

                # Encode to Opus at 24kHz
                try:
                    opus_frame = self.opus_encoder.encode(frame_bytes, self.output_frame_size)

                    if self.audio_frames_sent == 0:
                        logger.bind(tag=TAG).info(f"First Opus frame encoded: {len(opus_frame)} bytes")

                except Exception as e:
                    logger.bind(tag=TAG).error(f"Opus encoding failed: {e}")
                    continue

                # Send to client
                if self.conn.websocket:
                    if self.audio_frames_sent == 0:
                        logger.bind(tag=TAG).info(f"Sending first audio frame ({len(opus_frame)} bytes Opus) to ESP32")

                    await self.conn.send_audio_to_client(opus_frame)
                    self.audio_frames_sent += 1

                    if self.audio_frames_sent == 1:
                        logger.bind(tag=TAG).info(f"First audio frame sent successfully")
                    elif self.audio_frames_sent % 10 == 0:
                        logger.bind(tag=TAG).debug(f"Sent {self.audio_frames_sent} audio frames to device")

                    # Small yield to allow other tasks to run, but don't delay audio
                    # (60ms frames should be sent as fast as possible to minimize latency)
                    await asyncio.sleep(0.001)  # 1ms yield instead of 50ms delay
                else:
                    logger.bind(tag=TAG).error(f"Cannot send audio - websocket is None!")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling audio output: {e}")

    async def _handle_user_message(self, event: Dict[str, Any]):
        """Handle user's transcribed message"""
        try:
            message = event.get("message", {})
            content = message.get("content", "")

            if content:
                logger.bind(tag=TAG).info(f"User said: {content}")

                # Send transcription to client for display
                await self._send_transcription_to_client(content)

                # Report to management API
                enqueue_asr_report(self.conn, content, None)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling user message: {e}")

    async def _handle_assistant_message(self, event: Dict[str, Any]):
        """Handle assistant's text response"""
        try:
            message = event.get("message", {})
            content = message.get("content", "")

            if content:
                logger.bind(tag=TAG).info(f"Assistant said: {content}")

                # Report to management API
                enqueue_tts_report(self.conn, content, None)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling assistant message: {e}")

    async def _handle_user_interruption(self, event: Dict[str, Any]):
        """Handle user interruption"""
        try:
            logger.bind(tag=TAG).info("User interruption detected")

            # Stop current audio playback
            self.response_in_progress = False
            self.pcm_buffer = bytearray()

            # Send stop signal to client
            await self._send_audio_complete()

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling user interruption: {e}")

    async def _handle_assistant_end(self, event: Dict[str, Any]):
        """Handle assistant end of response"""
        try:
            logger.bind(tag=TAG).info(f"Assistant response complete - sent {self.audio_frames_sent} frames")

            # Flush any remaining audio
            await self._flush_audio_buffer()

            # Send stop signal
            await self._send_audio_complete()

            self.response_in_progress = False

            # Check if we should close connection (e.g., from handle_exit_intent)
            if hasattr(self.conn, 'close_after_chat') and self.conn.close_after_chat:
                logger.bind(tag=TAG).info("Closing connection after chat as requested")
                await self.conn.close()

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling assistant end: {e}")

    async def _handle_error(self, event: Dict[str, Any]):
        """Handle error from Hume EVI"""
        error_msg = event.get("message", "Unknown error")
        error_code = event.get("code", "")
        logger.bind(tag=TAG).error(f"Hume EVI error [{error_code}]: {error_msg}")

    async def _handle_tool_call(self, event: Dict[str, Any]):
        """Handle tool/function call from Hume EVI

        Event format:
        {
            "type": "tool_call",
            "tool_type": "function",
            "tool_call_id": "call_xyz123",
            "name": "get_weather",
            "parameters": "{\"location\":\"New York\"}"
        }
        """
        try:
            tool_call_id = event.get("tool_call_id")
            function_name = event.get("name")
            parameters_str = event.get("parameters", "{}")

            logger.bind(tag=TAG).info(f"Tool call received: {function_name} | ID: {tool_call_id}")

            # Parse parameters
            try:
                parameters = json.loads(parameters_str)
            except json.JSONDecodeError:
                parameters = {}
                logger.bind(tag=TAG).warning(f"Failed to parse tool parameters: {parameters_str}")

            # Execute function via UnifiedToolHandler
            if hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                result = await self.conn.func_handler.handle_llm_function_call(
                    self.conn,
                    {
                        "name": function_name,
                        "arguments": parameters
                    }
                )

                # Send result back to Hume EVI
                await self._send_tool_response(tool_call_id, result, function_name)
            else:
                logger.bind(tag=TAG).error("No function handler available")
                await self._send_tool_error(tool_call_id, "Function handler not available")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling tool call: {e}")
            await self._send_tool_error(tool_call_id, str(e))

    async def _send_tool_response(self, tool_call_id: str, result, function_name: str = ""):
        """Send tool response back to Hume EVI"""
        try:
            # Format result
            content = ""
            if hasattr(result, 'response') and result.response:
                content = result.response
            elif hasattr(result, 'result') and result.result:
                content = result.result
            else:
                content = str(result)

            # Send tool response message
            message = {
                "type": "tool_response",
                "tool_call_id": tool_call_id,
                "content": content
            }

            await self.ws.send(json.dumps(message))
            logger.bind(tag=TAG).info(f"Tool response sent | ID: {tool_call_id} | Function: {function_name}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending tool response: {e}")

    async def _send_tool_error(self, tool_call_id: str, error: str):
        """Send tool error to Hume EVI"""
        try:
            message = {
                "type": "tool_error",
                "tool_call_id": tool_call_id,
                "error": error,
                "content": f"Error executing tool: {error}"
            }
            await self.ws.send(json.dumps(message))
            logger.bind(tag=TAG).error(f"Tool error sent | ID: {tool_call_id} | Error: {error}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending tool error: {e}")

    async def _flush_audio_buffer(self):
        """Flush any remaining audio in the buffer"""
        try:
            if len(self.pcm_buffer) > 0:
                FRAME_SIZE_BYTES = self.output_frame_size * 2

                # Pad buffer to complete frame size
                padding_needed = FRAME_SIZE_BYTES - len(self.pcm_buffer)
                if padding_needed > 0:
                    self.pcm_buffer.extend(bytes(padding_needed))

                # Send the final frame at 24kHz
                frame_bytes = bytes(self.pcm_buffer[:FRAME_SIZE_BYTES])
                opus_frame = self.opus_encoder.encode(frame_bytes, self.output_frame_size)

                if self.conn.websocket:
                    await self.conn.send_audio_to_client(opus_frame)
                    logger.bind(tag=TAG).debug("Final audio frame sent")

                # Clear buffer
                self.pcm_buffer = bytearray()
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error flushing audio buffer: {e}")

    async def receive_audio(self, audio: bytes):
        """Receive audio from client and send to Hume EVI

        Args:
            audio: Opus-encoded audio frame from client at 16kHz
        """
        try:
            if not self.is_connected or not self.ws:
                logger.bind(tag=TAG).warning(f"Cannot receive audio - not connected")
                return

            # Skip if music is playing
            if self.is_music_playing:
                return

            # Update activity timestamp
            import time
            self.last_activity_time = time.time()
            self.conn.last_activity_time = self.last_activity_time * 1000

            # Log every 50th frame
            self.audio_frames_received += 1
            if self.audio_frames_received % 50 == 0:
                logger.bind(tag=TAG).info(f"Received {self.audio_frames_received} audio frames from client")

            # Decode Opus to PCM16 at 16kHz (960 samples = 60ms)
            try:
                pcm_16khz = self.opus_decoder.decode(audio, self.input_frame_size)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Opus decode error: {e}")
                return

            # Send 16kHz PCM directly to Hume (no resampling needed)
            # Encode PCM16 to base64 for Hume EVI
            audio_base64 = base64.b64encode(pcm_16khz).decode('utf-8')

            # Send to Hume EVI
            message = {
                "type": "audio_input",
                "data": audio_base64
            }

            await self.ws.send(json.dumps(message))

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error receiving audio: {e}")

    async def _send_tts_initial_start(self):
        """Send TTS initial start signal to client (enables codec)"""
        try:
            if self.conn.websocket:
                tts_msg = {
                    "type": "tts",
                    "state": "start",
                    "session_id": self.conn.session_id,
                    "volume_control": 50  # Set volume to 50% to prevent AEC feedback/echo
                }
                await self.conn.websocket.send(json.dumps(tts_msg))
                logger.bind(tag=TAG).info(f"Sent TTS initial start signal with volume_control=50: {tts_msg}")
            else:
                logger.bind(tag=TAG).error("Cannot send TTS initial start - websocket is None!")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS initial start: {e}")

    async def _send_tts_start(self):
        """Send TTS start signal to client"""
        try:
            if self.conn.websocket:
                tts_msg = {
                    "type": "tts",
                    "state": "sentence_start",
                    "text": "",
                    "session_id": self.conn.session_id
                }
                await self.conn.websocket.send(json.dumps(tts_msg))
                logger.bind(tag=TAG).info(f"Sent TTS sentence_start signal to client: {tts_msg}")
            else:
                logger.bind(tag=TAG).error("Cannot send TTS start - websocket is None!")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS start: {e}")

    async def _send_audio_complete(self):
        """Send audio complete signal to client"""
        try:
            if self.conn.websocket:
                tts_stop_msg = {
                    "type": "tts",
                    "state": "stop",
                    "session_id": self.conn.session_id
                }
                await self.conn.websocket.send(json.dumps(tts_stop_msg))
                logger.bind(tag=TAG).info(f"Sent TTS stop signal to client: {tts_stop_msg}")
            else:
                logger.bind(tag=TAG).error("Cannot send TTS stop - websocket is None!")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending audio complete: {e}")

    async def _send_transcription_to_client(self, text: str):
        """Send transcription text to client for display"""
        try:
            if self.conn.websocket:
                await self.conn.websocket.send(
                    json.dumps({
                        "type": "tts",
                        "state": "sentence_start",
                        "text": text,
                        "session_id": self.conn.session_id
                    })
                )
                logger.bind(tag=TAG).debug(f"Sent transcription to client: {text}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending transcription: {e}")

    async def update_tools(self):
        """Update Hume config with latest tools (called when device MCP tools become available)

        Note: For Hume EVI, tool configuration is managed through the web console config.
        Dynamic tool updates are not supported via REST API for existing configs.
        Users should configure tools in the Hume console when creating the config.
        """
        try:
            if not self.config_id:
                logger.bind(tag=TAG).warning("Cannot update tools - no config_id available")
                return

            # Hume configs are managed through web console - tools cannot be dynamically updated
            # Log the tools for informational purposes only
            if hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                tools = self.conn.func_handler.get_functions()
                tool_names = [t.get("function", {}).get("name") for t in tools]
                logger.bind(tag=TAG).info(
                    f"Device MCP tools registered ({len(tools)} tools): {tool_names}. "
                    f"Note: Hume config {self.config_id} must be updated manually via web console to use new tools."
                )


        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to update tools: {e}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            self.is_connected = False

            # Cancel receive task
            if self.receive_task and not self.receive_task.done():
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass

            # Close WebSocket
            if self.ws:
                await self.ws.close()
                self.ws = None

            logger.bind(tag=TAG).info("Hume EVI provider cleaned up")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during cleanup: {e}")
