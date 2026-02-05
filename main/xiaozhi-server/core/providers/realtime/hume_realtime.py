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
        # Hume EVI accepts linear16 PCM at various sample rates
        # We'll use 16kHz to match ESP32 output
        self.sample_rate = 16000  # 16kHz matches ESP32
        self.channels = 1  # Mono
        self.audio_format = "linear16"  # 16-bit PCM

        # Opus decoder for client audio at 16kHz
        self.opus_decoder = opuslib_next.Decoder(16000, 1)

        # Opus encoder for sending audio to client at 16kHz
        self.opus_encoder = opuslib_next.Encoder(16000, 1, opuslib_next.APPLICATION_VOIP)

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
        base_url = "wss://api.hume.ai/v0/assistant/chat"
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
            # Send session settings for audio format
            session_settings = {
                "type": "session_settings",
                "audio": {
                    "encoding": self.audio_format,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels
                }
            }

            await self.ws.send(json.dumps(session_settings))
            logger.bind(tag=TAG).info(f"Session configured | Audio: {self.audio_format}@{self.sample_rate}Hz")

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
            # Mark response as in progress
            if not self.response_in_progress:
                self.response_in_progress = True
                self.response_start_time = 0
                await self._send_tts_start()
                logger.bind(tag=TAG).info("Assistant response started")

            # Get base64 encoded audio data
            audio_data_base64 = event.get("data", "")
            if not audio_data_base64:
                return

            # Decode from base64 to get WAV file
            audio_wav = base64.b64decode(audio_data_base64)

            # Skip WAV header (first 44 bytes) to get raw PCM16 data
            # WAV header is 44 bytes for standard PCM format
            pcm_data = audio_wav[44:] if len(audio_wav) > 44 else audio_wav

            # Add to buffer
            self.pcm_buffer.extend(pcm_data)

            # Send complete frames to client (960 samples = 60ms at 16kHz)
            FRAME_SIZE_SAMPLES = 960
            FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2  # 16-bit = 2 bytes per sample

            while len(self.pcm_buffer) >= FRAME_SIZE_BYTES:
                # Extract one frame
                frame_bytes = bytes(self.pcm_buffer[:FRAME_SIZE_BYTES])
                self.pcm_buffer = self.pcm_buffer[FRAME_SIZE_BYTES:]

                # Encode to Opus
                opus_frame = self.opus_encoder.encode(frame_bytes, FRAME_SIZE_SAMPLES)

                # Send to client
                if self.conn.websocket:
                    await self.conn.send_audio_to_client(opus_frame)
                    self.audio_frames_sent += 1
                    # Pace sending to prevent buffer overflow
                    await asyncio.sleep(0.050)  # 50ms delay

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
                FRAME_SIZE_SAMPLES = 960
                FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2

                # Pad buffer to complete frame size
                padding_needed = FRAME_SIZE_BYTES - len(self.pcm_buffer)
                if padding_needed > 0:
                    self.pcm_buffer.extend(bytes(padding_needed))

                # Send the final frame
                frame_bytes = bytes(self.pcm_buffer[:FRAME_SIZE_BYTES])
                opus_frame = self.opus_encoder.encode(frame_bytes, 960)

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
                pcm_16khz = self.opus_decoder.decode(audio, 960)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Opus decode error: {e}")
                return

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

    async def _send_tts_start(self):
        """Send TTS start signal to client"""
        try:
            if self.conn.websocket:
                await self.conn.websocket.send(
                    json.dumps({
                        "type": "tts",
                        "state": "sentence_start",
                        "text": "",
                        "session_id": self.conn.session_id
                    })
                )
                logger.bind(tag=TAG).debug("Sent TTS start signal to client")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS start: {e}")

    async def _send_audio_complete(self):
        """Send audio complete signal to client"""
        try:
            if self.conn.websocket:
                await self.conn.websocket.send(
                    json.dumps({
                        "type": "tts",
                        "state": "stop",
                        "session_id": self.conn.session_id
                    })
                )
                logger.bind(tag=TAG).debug("Sent TTS stop signal to client")
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
