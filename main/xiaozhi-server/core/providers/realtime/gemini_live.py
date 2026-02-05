"""Gemini Live API Provider

This provider integrates Google's Gemini Live API which combines ASR, LLM, and TTS
in a single low-latency bidirectional streaming connection. It intercepts function calls
and executes them server-side using the existing UnifiedToolHandler.

Architecture:
    Audio Input → Gemini Live API (ASR+LLM) → Function Call Detection
                                           ↓
                                  Execute via UnifiedToolHandler
                                           ↓
                                  Return to Gemini Live API
                                           ↓
                                  TTS Output → Audio

Key advantages over OpenAI Realtime:
- Better echo cancellation and VAD handling
- Lower latency in some regions
- More flexible turn detection options
"""

import json
import base64
import asyncio
import os
import opuslib_next
from typing import Dict, Any, Optional
from config.logger import setup_logging
from google import genai
from google.genai import types

TAG = __name__
logger = setup_logging()


class GeminiLiveProvider:
    """Gemini Live API Provider

    Combines ASR, LLM, and TTS in a single streaming connection for ultra-low latency.
    Integrates with existing tool execution infrastructure.
    """

    def __init__(self, config: Dict[str, Any], conn):
        """Initialize Gemini Live provider

        Args:
            config: Provider configuration containing:
                - use_vertex: Whether to use Vertex AI (required for Live API)
                - project_id: Google Cloud project ID (for Vertex AI)
                - location: Google Cloud location (for Vertex AI, default: us-central1)
                - api_key: Google API key (not used for Live API, for future compatibility)
                - model: Model name (default: gemini-2.0-flash-exp)
                - voice: Voice name (default: Puck)
                - language: Language code (default: en)
                - instructions: System instructions/prompt
                - temperature: Model temperature (default: 0.8)
            conn: Connection handler instance
        """
        self.conn = conn
        self.config = config

        # Vertex AI configuration (required for Gemini Live)
        self.use_vertex = config.get("use_vertex", True)
        self.project_id = config.get("project_id")
        self.location = config.get("location", "us-central1")

        # Try to get project_id from environment if not in config
        if self.use_vertex and not self.project_id:
            import os
            self.project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")

            if not self.project_id:
                raise ValueError(
                    "Google Cloud project_id is required for Gemini Live (Vertex AI). "
                    "Set it in config or use GCP_PROJECT_ID environment variable."
                )

        # Model configuration
        self.model = config.get("model", "gemini-2.0-flash-exp")
        self.voice = config.get("voice", "Puck")  # Puck, Charon, Kore, Fenrir, Aoede
        self.language = config.get("language", "en")

        # Ensure temperature is float
        temp = config.get("temperature", 0.8)
        self.temperature = float(temp) if temp else 0.8

        # Audio configuration
        # Gemini Live supports PCM16 at 16kHz - perfect match for ESP32!
        self.sample_rate = 16000  # 16kHz - no resampling needed!

        # Opus encoder/decoder for communication with client at 16kHz
        self.opus_encoder = opuslib_next.Encoder(16000, 1, opuslib_next.APPLICATION_VOIP)
        self.opus_decoder = opuslib_next.Decoder(16000, 1)

        # Gemini client
        self.client = None
        self.session = None
        self._session_context = None  # Store the async context manager

        # Session state
        self.is_connected = False
        self.is_processing = False

        # Tasks
        self.receive_task = None
        self.send_task = None

        # Audio queues for async processing
        self.audio_input_queue = asyncio.Queue()

        # Audio buffers
        self.pcm_buffer = bytearray()  # Buffer for accumulating PCM16 samples
        self.audio_frames_sent = 0
        self.audio_frames_received = 0  # Counter for received audio chunks

        # Conversation state
        self.response_in_progress = False

        # Transcription tracking
        self.current_transcription = ""
        self.transcription_sent = False

        # Activity tracking for manual VAD control
        self.user_activity_active = False
        self.audio_frames_since_turn = 0

        # Function call handling
        self.pending_function_calls = {}

        logger.bind(tag=TAG).info(
            f"Gemini Live provider initialized | "
            f"Model: {self.model} | Voice: {self.voice} | Language: {self.language} | "
            f"Vertex AI: {self.use_vertex}"
        )

    async def connect(self):
        """Connect to Gemini Live API"""
        try:
            logger.bind(tag=TAG).info(f"Connecting to Gemini Live API: {self.model}")

            # Initialize Gemini client
            if self.use_vertex:
                # Use Vertex AI (required for Gemini Live)
                # Check for credentials
                creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                if creds_path:
                    logger.bind(tag=TAG).info(f"Using credentials from: {creds_path}")
                    if not os.path.exists(creds_path):
                        logger.bind(tag=TAG).warning(f"Credentials file not found at: {creds_path}")
                else:
                    logger.bind(tag=TAG).info("Using Application Default Credentials (ADC)")

                self.client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                logger.bind(tag=TAG).info(f"Using Vertex AI | Project: {self.project_id} | Location: {self.location}")
            else:
                # Use Developer API (not supported for Live API yet)
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ValueError("API key required for Developer API")
                self.client = genai.Client(api_key=api_key)
                logger.bind(tag=TAG).info("Using Developer API")

            # Get system instructions from connection
            instructions = await self._get_system_instructions()

            # Get tools from UnifiedToolHandler
            tools = await self._get_tools()

            # Configure session
            config_dict = {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": self.voice
                        }
                    }
                },
                # Enable proactivity for better conversation flow
                "proactivity": {
                    "proactive_audio": True
                },
                # Enable transcription for debugging
                "input_audio_transcription": {},  # Transcribe user audio
                "output_audio_transcription": {}  # Transcribe bot audio
            }

            # Add system instruction if available
            if instructions:
                config_dict["system_instruction"] = {
                    "parts": [{"text": instructions}]
                }

            # Add tools if available
            if tools:
                config_dict["tools"] = tools

            config = types.LiveConnectConfig(**config_dict)

            # Start the session loop as a background task
            # This runs the entire session inside async with
            self.session_task = asyncio.create_task(self._run_session(config))

            # Wait a bit for connection to establish
            await asyncio.sleep(0.1)

            self.is_connected = True
            logger.bind(tag=TAG).info("Gemini Live session started")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error connecting to Gemini Live: {e}", exc_info=True)
            self.is_connected = False

    async def _get_system_instructions(self) -> Optional[str]:
        """Get system instructions from connection's prompt manager"""
        try:
            if hasattr(self.conn, 'prompt_manager') and self.conn.prompt_manager:
                # Get device_id from connection headers
                device_id = self.conn.headers.get("device-id", "unknown")
                # Build enhanced prompt with empty user_prompt (for system instructions)
                prompt = self.conn.prompt_manager.build_enhanced_prompt(
                    user_prompt="",
                    device_id=device_id
                )
                return prompt
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error getting system instructions: {e}")
        return None

    async def _get_tools(self) -> Optional[list]:
        """Get tools from UnifiedToolHandler in Gemini format"""
        try:
            if not hasattr(self.conn, 'tool_handler') or not self.conn.tool_handler:
                return None

            # Get OpenAI format tools
            openai_tools = self.conn.tool_handler.get_tools()
            if not openai_tools:
                return None

            # Convert to Gemini format
            gemini_tools = []
            for tool in openai_tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    gemini_tool = types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=func.get("name"),
                                description=func.get("description"),
                                parameters=func.get("parameters")
                            )
                        ]
                    )
                    gemini_tools.append(gemini_tool)

            logger.bind(tag=TAG).info(f"Configured {len(gemini_tools)} tools for Gemini Live")
            return gemini_tools

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error getting tools: {e}")
            return None

    async def _run_session(self, config):
        """Run the Gemini Live session inside async with context"""
        try:
            logger.bind(tag=TAG).info("Starting Gemini Live session with async with")

            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                self.session = session
                logger.bind(tag=TAG).info("Connected to Gemini Live API")

                # Start send and receive loops
                send_task = asyncio.create_task(self._send_audio_loop(session))
                receive_task = asyncio.create_task(self._receive_loop(session))

                # Wait for both tasks (they run until cancelled or error)
                await asyncio.gather(send_task, receive_task, return_exceptions=True)

        except asyncio.CancelledError:
            logger.bind(tag=TAG).info("Session cancelled")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in session: {e}", exc_info=True)
        finally:
            self.is_connected = False
            self.session = None
            logger.bind(tag=TAG).info("Gemini Live session ended")

    async def _send_audio_loop(self, session):
        """Send audio from queue to Gemini Live"""
        try:
            logger.bind(tag=TAG).info("Started audio send loop")
            while self.is_connected:
                # Get audio chunk from queue
                chunk = await self.audio_input_queue.get()
                if chunk is None:  # Poison pill to stop loop
                    break

                # Send to Gemini Live using send_realtime_input() (only available in async with)
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type=f"audio/pcm;rate={self.sample_rate}"
                    )
                )
                self.audio_frames_received += 1
                if self.audio_frames_received % 50 == 0:
                    logger.bind(tag=TAG).info(f"Sent {self.audio_frames_received} audio frames to Gemini Live")
        except asyncio.CancelledError:
            logger.bind(tag=TAG).info("Audio send loop cancelled")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in send loop: {e}", exc_info=True)
            self.is_connected = False

    async def _receive_loop(self, session):
        """Receive and process messages from Gemini Live"""
        try:
            logger.bind(tag=TAG).info("Started receiving from Gemini Live")
            response_count = 0
            async for response in session.receive():
                response_count += 1
                logger.bind(tag=TAG).debug(f"Received response #{response_count} from Gemini: {type(response)}")
                await self._handle_response(response)
        except asyncio.CancelledError:
            logger.bind(tag=TAG).info("Receive loop cancelled")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in receive loop: {e}", exc_info=True)
            self.is_connected = False

    async def _handle_response(self, response):
        """Handle response from Gemini Live API"""
        try:
            # Log all response attributes for debugging
            has_server_content = hasattr(response, 'server_content') and response.server_content
            has_tool_call = hasattr(response, 'tool_call') and response.tool_call
            has_setup_complete = hasattr(response, 'setup_complete') and response.setup_complete

            if not has_server_content and not has_tool_call and not has_setup_complete:
                logger.bind(tag=TAG).debug(f"Received response with no actionable content: {dir(response)}")
                return

            # Setup complete
            if has_setup_complete:
                logger.bind(tag=TAG).info("Gemini Live setup complete")
                return

            # Server content - audio or text response
            if response.server_content:
                server_content = response.server_content

                # Input transcription (user speech)
                if server_content.input_transcription and server_content.input_transcription.text:
                    logger.bind(tag=TAG).info(f"User said: {server_content.input_transcription.text}")

                # Output transcription (bot speech)
                if server_content.output_transcription and server_content.output_transcription.text:
                    transcription = server_content.output_transcription.text
                    logger.bind(tag=TAG).info(f"Bot said: {transcription}")

                    # Accumulate transcription
                    self.current_transcription += transcription

                    # Send transcription to client if not already sent
                    if not self.transcription_sent and self.current_transcription.strip():
                        await self._send_transcription_to_client(self.current_transcription)
                        self.transcription_sent = True

                # Model turn with audio/text parts
                if server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        # Audio response
                        if hasattr(part, 'inline_data') and part.inline_data:
                            logger.bind(tag=TAG).debug(f"Received audio data ({len(part.inline_data.data)} bytes)")
                            await self._handle_audio_response(part.inline_data.data)

                        # Text response (for debugging)
                        elif hasattr(part, 'text') and part.text:
                            logger.bind(tag=TAG).info(f"Bot text: {part.text}")

                # Turn complete
                if server_content.turn_complete:
                    logger.bind(tag=TAG).info("Turn complete")
                    await self._handle_turn_complete()

                # Interrupted
                if server_content.interrupted:
                    logger.bind(tag=TAG).info("Turn interrupted by user")

            # Tool call
            if response.tool_call:
                logger.bind(tag=TAG).info("Received tool_call from Gemini")
                await self._handle_tool_call(response.tool_call)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling response: {e}", exc_info=True)

    async def _handle_audio_response(self, audio_data: bytes):
        """Handle audio response from Gemini Live

        Gemini sends PCM16 at 16kHz - same as ESP32, so no resampling needed!
        """
        try:
            # Send TTS start signal on first audio chunk
            if not self.response_in_progress:
                self.response_in_progress = True
                logger.bind(tag=TAG).info("Bot started speaking")
                await self._send_tts_start()

            # Add to buffer
            self.pcm_buffer.extend(audio_data)
            logger.bind(tag=TAG).info(f"Received {len(audio_data)} bytes, buffer now {len(self.pcm_buffer)} bytes")

            # Process buffer in chunks of 960 samples (1920 bytes for PCM16)
            # 960 samples = 60ms at 16kHz - perfect for Opus
            FRAME_SIZE_SAMPLES = 960
            FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2  # 2 bytes per sample (int16)

            frames_sent_this_batch = 0
            while len(self.pcm_buffer) >= FRAME_SIZE_BYTES:
                # Extract one frame
                frame_bytes = bytes(self.pcm_buffer[:FRAME_SIZE_BYTES])
                self.pcm_buffer = self.pcm_buffer[FRAME_SIZE_BYTES:]

                # Encode to Opus at 16kHz
                opus_frame = self.opus_encoder.encode(frame_bytes, FRAME_SIZE_SAMPLES)

                # Send to client
                if self.conn.websocket:
                    await self.conn.send_audio_to_client(opus_frame)
                    self.audio_frames_sent += 1
                    frames_sent_this_batch += 1
                else:
                    logger.bind(tag=TAG).error("No websocket connection to send audio")
                    break

            if frames_sent_this_batch > 0:
                logger.bind(tag=TAG).info(f"Sent {frames_sent_this_batch} frames to client (total: {self.audio_frames_sent})")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling audio response: {e}", exc_info=True)

    async def _handle_tool_call(self, tool_call):
        """Handle tool call from Gemini Live"""
        try:
            for fc in tool_call.function_calls:
                function_name = fc.name
                call_id = fc.id
                arguments = dict(fc.args) if fc.args else {}

                logger.bind(tag=TAG).info(f"Executing function: {function_name} | Call ID: {call_id}")

                # Execute via UnifiedToolHandler
                if hasattr(self.conn, 'tool_handler') and self.conn.tool_handler:
                    result = await self.conn.tool_handler.execute_tool(
                        function_name=function_name,
                        arguments=arguments,
                        conn=self.conn
                    )

                    # Send result back to Gemini
                    await self.session.send(
                        types.LiveClientToolResponse(
                            function_responses=[
                                types.FunctionResponse(
                                    id=call_id,
                                    name=function_name,
                                    response={"result": result}
                                )
                            ]
                        )
                    )

                    logger.bind(tag=TAG).info(f"Function result sent | Call ID: {call_id}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling tool call: {e}", exc_info=True)

    async def _handle_turn_complete(self):
        """Handle turn completion (bot finished speaking)"""
        try:
            # Flush any remaining audio
            await self._flush_audio_buffer()

            # Send TTS stop signal
            await self._send_tts_stop()

            logger.bind(tag=TAG).info(f"Turn complete - sent {self.audio_frames_sent} audio frames")
            logger.bind(tag=TAG).info(f"Full transcription: {self.current_transcription}")

            self.audio_frames_sent = 0

            # Reset transcription state for next turn
            self.current_transcription = ""
            self.transcription_sent = False

            # Reset response state
            self.response_in_progress = False

            # Reset activity tracking for next user turn
            self.user_activity_active = False
            self.audio_frames_since_turn = 0

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling turn complete: {e}", exc_info=True)

    async def _flush_audio_buffer(self):
        """Flush any remaining audio in the buffer"""
        try:
            if len(self.pcm_buffer) > 0:
                logger.bind(tag=TAG).info(f"Flushing {len(self.pcm_buffer)} bytes of remaining audio")

                # Pad to frame size if needed
                FRAME_SIZE_BYTES = 960 * 2
                if len(self.pcm_buffer) < FRAME_SIZE_BYTES:
                    padding_needed = FRAME_SIZE_BYTES - len(self.pcm_buffer)
                    self.pcm_buffer.extend(bytes(padding_needed))
                    logger.bind(tag=TAG).info(f"Padded buffer with {padding_needed} bytes")

                # Send final frame
                frame_bytes = bytes(self.pcm_buffer[:FRAME_SIZE_BYTES])
                opus_frame = self.opus_encoder.encode(frame_bytes, 960)

                if self.conn.websocket:
                    await self.conn.send_audio_to_client(opus_frame)
                    self.audio_frames_sent += 1
                    logger.bind(tag=TAG).info(f"Final audio frame sent (frame #{self.audio_frames_sent})")
                else:
                    logger.bind(tag=TAG).error("Cannot flush - no websocket connection")

                # Clear buffer
                self.pcm_buffer = bytearray()
            else:
                logger.bind(tag=TAG).info("No remaining audio to flush")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error flushing audio buffer: {e}", exc_info=True)

    async def receive_audio(self, audio: bytes):
        """Receive audio from client and queue for sending to Gemini Live

        Streams audio continuously - Gemini Live handles turn detection automatically.

        Args:
            audio: Opus-encoded audio packet from ESP32 at 16kHz
        """
        try:
            if not self.is_connected:
                return

            # Decode Opus to PCM16 at 16kHz
            pcm_16khz = self.opus_decoder.decode(audio, 960)

            # Queue audio - Gemini's automatic VAD and proactivity handles everything
            await self.audio_input_queue.put(pcm_16khz)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error receiving audio: {e}", exc_info=True)

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
                logger.bind(tag=TAG).info(f"Sent transcription to client: {text}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending transcription: {e}")

    async def _send_tts_start(self):
        """Signal to client that audio output is starting"""
        try:
            if self.conn.websocket:
                await self.conn.websocket.send(
                    json.dumps({
                        "type": "tts",
                        "state": "start",
                        "session_id": self.conn.session_id
                    })
                )
                logger.bind(tag=TAG).debug("Sent TTS start signal")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS start: {e}")

    async def _send_tts_stop(self):
        """Signal to client that audio output is complete"""
        try:
            if self.conn.websocket:
                await self.conn.websocket.send(
                    json.dumps({
                        "type": "tts",
                        "state": "stop",
                        "session_id": self.conn.session_id
                    })
                )
                logger.bind(tag=TAG).debug("Sent TTS stop signal")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS stop: {e}")

    async def disconnect(self):
        """Disconnect from Gemini Live API"""
        try:
            self.is_connected = False

            # Send poison pill to stop send loop
            await self.audio_input_queue.put(None)

            # Cancel the session task (which will close the async with block)
            if hasattr(self, 'session_task') and self.session_task and not self.session_task.done():
                self.session_task.cancel()
                try:
                    await self.session_task
                except asyncio.CancelledError:
                    pass

            logger.bind(tag=TAG).info("Gemini Live provider disconnected")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error disconnecting: {e}")
