"""OpenAI Realtime API Provider

This provider integrates OpenAI's Realtime API which combines ASR, LLM, and TTS
in a single low-latency WebSocket connection. It intercepts function calls
and executes them server-side using the existing UnifiedToolHandler.

Architecture:
    Audio Input → Realtime API (ASR+LLM) → Function Call Detection
                                           ↓
                                  Execute via UnifiedToolHandler
                                           ↓
                                  Return to Realtime API
                                           ↓
                                  TTS Output → Audio

This eliminates the ASR→LLM→TTS pipeline latency while preserving
all existing functionality (MCP, plugins, knowledge bases, custom prompts).
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

TAG = __name__
logger = setup_logging()


class OpenAIRealtimeProvider:
    """OpenAI Realtime API Provider

    Combines ASR, LLM, and TTS in a single WebSocket connection for ultra-low latency.
    Integrates with existing tool execution infrastructure.
    """

    def __init__(self, config: Dict[str, Any], conn):
        """Initialize OpenAI Realtime provider

        Args:
            config: Provider configuration containing:
                - api_key: OpenAI API key
                - model: Model name (default: gpt-4o-realtime-preview-2024-12-17)
                - voice: Voice name (default: alloy)
                - language: Language code (default: en)
                - instructions: System instructions/prompt
                - temperature: Model temperature (default: 0.8)
                - max_response_output_tokens: Max tokens per response
            conn: Connection handler instance
        """
        self.conn = conn
        self.config = config

        # OpenAI API configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for Realtime provider")

        # Model configuration
        self.model = config.get("model", "gpt-4o-realtime-preview-2024-12-17")
        self.voice = config.get("voice", "alloy")  # alloy, echo, shimmer
        self.language = config.get("language", "en")
        self.voice_prompt = config.get("voice_prompt", "")  # Voice customization instructions
        # Ensure temperature is float, not string
        temp = config.get("temperature", 0.8)
        self.temperature = float(temp) if temp else 0.8
        # Ensure max_response_tokens is int, not string
        max_tokens = config.get("max_response_output_tokens", 4096)
        self.max_response_tokens = int(max_tokens) if max_tokens and str(max_tokens) != "inf" else 4096

        # Audio configuration
        self.input_audio_format = "pcm16"  # 16-bit PCM
        self.output_audio_format = "pcm16"
        self.sample_rate = 24000  # Realtime API uses 24kHz

        # Opus encoder/decoder for communication with client at 16kHz
        # ESP32 firmware is fixed at 16kHz, so we must resample between 16kHz and 24kHz
        self.opus_encoder = opuslib_next.Encoder(16000, 1, opuslib_next.APPLICATION_VOIP)
        self.opus_decoder = opuslib_next.Decoder(16000, 1)

        # WebSocket connection
        self.ws = None
        self.ws_url = f"wss://api.openai.com/v1/realtime?model={self.model}"

        # Session state
        self.session_id = None
        self.is_connected = False
        self.is_processing = False
        self.is_music_playing = False  # Flag to pause audio processing during music playback

        # Tasks
        self.receive_task = None
        self.send_task = None

        # Audio buffers
        self.audio_queue = asyncio.Queue()
        self.output_audio_buffer = []
        self.pcm_buffer = bytearray()  # Output buffer (OpenAI → Client) at 16kHz
        self._input_pcm_buffer_24khz = bytearray()  # Input buffer (Client → OpenAI) at 24kHz for accumulation

        # Conversation state
        self.conversation_id = None
        self.response_in_progress = False
        self.audio_frames_sent = 0  # Track audio frames sent to device
        self.audio_frames_received = 0  # Track audio frames received from client

        # Audio debugging - dump received audio to files
        self.debug_audio_enabled = False  # Disable audio dumping (set to True for debugging)
        self.debug_pcm_buffer = bytearray()  # Buffer for decoded PCM data at 16kHz
        self.debug_audio_start_time = None

        # Keepalive tracking
        import time
        self.last_activity_time = time.time()  # Track last message received

        # Interruption handling
        self.last_speech_started_time = 0  # Track when speech was detected
        self.response_start_time = 0  # Track when response started

        # Function call handling
        self.pending_function_calls = {}

        # Voiceprint (speaker identification) state
        self.voiceprint_enabled = False
        self.voiceprint_buffer = b""  # Buffer PCM audio for voiceprint identification
        self.voiceprint_buffer_duration = 0.0  # Duration in seconds
        self.voiceprint_min_duration = 3.0  # Minimum 3 seconds of audio for identification
        self.voiceprint_max_duration = 5.0  # Maximum 5 seconds to buffer
        self.voiceprint_identified = False  # True after first successful identification
        self.current_speaker = None  # Current identified speaker name
        self.voiceprint_task = None  # Background task for identification

        # Audio buffering for chat history reporting (Opus packets)
        self.user_audio_buffer = []  # List of Opus packets for user speech (for chat history with audio)
        self.agent_audio_buffer = []  # List of Opus packets for agent speech (for chat history with audio)
        self.max_audio_buffer_duration = 3.5  # Maximum 3.5 seconds of audio to buffer
        self.opus_frame_duration_ms = 60  # 60ms Opus frames

        # Check if voiceprint is available from connection
        if hasattr(conn, 'voiceprint_provider') and conn.voiceprint_provider:
            self.voiceprint_enabled = True
            logger.bind(tag=TAG).info("Voiceprint integration enabled for OpenAI Realtime")

        logger.bind(tag=TAG).info(
            f"OpenAI Realtime provider initialized | "
            f"Model: {self.model} | Voice: {self.voice} | Language: {self.language} | "
            f"Voiceprint: {'Enabled' if self.voiceprint_enabled else 'Disabled'}"
        )

    async def connect(self):
        """Establish WebSocket connection to OpenAI Realtime API"""
        try:
            logger.bind(tag=TAG).info(f"Connecting to OpenAI Realtime API: {self.ws_url}")

            # Connect with authentication
            self.ws = await websockets.connect(
                self.ws_url,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                },
                max_size=10000000,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=30,   # Increased timeout for long conversations
                close_timeout=10,
            )

            self.session_id = str(uuid.uuid4())
            self.is_connected = True

            logger.bind(tag=TAG).info(f"Connected to OpenAI Realtime API | Session: {self.session_id}")

            # Configure session
            await self._configure_session()

            # Start receive task
            self.receive_task = asyncio.create_task(self._receive_loop())

            return True

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to connect to OpenAI Realtime API: {e}")
            self.is_connected = False
            return False

    async def _configure_session(self):
        """Configure the Realtime API session with instructions and settings"""
        try:
            # Get system instructions from prompt manager
            instructions = self.conn.prompt if hasattr(self.conn, 'prompt') else None
            if not instructions:
                instructions = self.config.get("instructions", "You are a helpful assistant.")

            # Prepend voice_prompt at the START of instructions so it's not buried
            # Use ALL CAPS for emphasis - OpenAI Realtime follows these more strictly
            if self.voice_prompt:
                voice_section = (
                    "## VOICE AND SPEECH STYLE - FOLLOW THESE RULES FOR HOW YOU SPEAK:\n"
                    f"{self.voice_prompt}\n"
                    "ALWAYS MAINTAIN THIS VOICE STYLE THROUGHOUT THE ENTIRE CONVERSATION.\n\n"
                )
                instructions = voice_section + instructions
                logger.bind(tag=TAG).info(f"Added voice customization prompt at start of instructions")

            # Add memory context if available
            if hasattr(self.conn, 'memory') and self.conn.memory:
                try:
                    memory_context = await self.conn.memory.query_memory("")
                    if memory_context:
                        memory_section = (
                            "\n\n## MEMORY - WHAT YOU REMEMBER ABOUT THIS USER:\n"
                            f"{memory_context}\n"
                            "USE THIS INFORMATION TO PERSONALIZE YOUR RESPONSES.\n"
                        )
                        instructions += memory_section
                        logger.bind(tag=TAG).info(f"Added memory context to instructions")
                except Exception as e:
                    logger.bind(tag=TAG).warning(f"Failed to add memory context: {e}")

            # Get available tools from function handler
            tools = []
            if hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                function_descriptions = self.conn.func_handler.get_functions()
                if function_descriptions:
                    # Convert to Realtime API tool format
                    for func_desc in function_descriptions:
                        func_info = func_desc.get("function", {})
                        tools.append({
                            "type": "function",
                            "name": func_info.get("name"),
                            "description": func_info.get("description"),
                            "parameters": func_info.get("parameters", {})
                        })

            # Session configuration
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": instructions,
                    "voice": self.voice,
                    "input_audio_format": self.input_audio_format,
                    "output_audio_format": self.output_audio_format,
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.7,  # Increased from 0.6 to reduce false triggers from background noise
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 1000,  # Increased from 700ms to 1000ms to reduce phantom speech detection
                        "create_response": True,  # Auto-create response after user speech
                        "interrupt_response": True  # Enable OpenAI's automatic interruption handling
                    },
                    "temperature": self.temperature,
                    "max_response_output_tokens": self.max_response_tokens,
                }
            }

            # Add tools if available
            if tools:
                session_config["session"]["tools"] = tools
                session_config["session"]["tool_choice"] = "auto"
                tool_names = [t.get("name") for t in tools]
                logger.bind(tag=TAG).info(f"Tools being sent to OpenAI: {tool_names}")

            await self.ws.send(json.dumps(session_config))
            logger.bind(tag=TAG).info(f"Session configured | Tools: {len(tools)} | Voice: {self.voice}")

            # Clear audio buffers and reset counters to ensure clean session start
            self.pcm_buffer = bytearray()
            self._input_pcm_buffer_24khz = bytearray()
            self.debug_pcm_buffer = bytearray()
            self.debug_audio_start_time = None
            self.audio_frames_sent = 0
            self.audio_frames_received = 0
            # Clear any residual audio in OpenAI's buffer
            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            logger.bind(tag=TAG).info("Audio buffers cleared for new session")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to configure session: {e}")

    async def update_tools(self):
        """Update session with latest tools (called when device MCP tools become available)"""
        try:
            if not self.is_connected or not self.ws:
                logger.bind(tag=TAG).warning("Cannot update tools - not connected")
                return

            # Get latest tools from function handler
            tools = []
            if hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                function_descriptions = self.conn.func_handler.get_functions()
                if function_descriptions:
                    # Convert to Realtime API tool format
                    for func_desc in function_descriptions:
                        func_info = func_desc.get("function", {})
                        tools.append({
                            "type": "function",
                            "name": func_info.get("name"),
                            "description": func_info.get("description"),
                            "parameters": func_info.get("parameters", {})
                        })

            if tools:
                # Send session update with new tools
                session_update = {
                    "type": "session.update",
                    "session": {
                        "tools": tools,
                        "tool_choice": "auto"
                    }
                }
                await self.ws.send(json.dumps(session_update))
                tool_names = [t.get("name") for t in tools]
                logger.bind(tag=TAG).info(f"Updated session with {len(tools)} tools: {tool_names}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to update tools: {e}")

    async def _receive_loop(self):
        """Main loop for receiving events from OpenAI Realtime API"""
        import time
        try:
            while self.is_connected and self.ws:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    event = json.loads(message)
                    await self._handle_event(event)

                except asyncio.TimeoutError:
                    # Check if we've been idle too long
                    idle_time = time.time() - self.last_activity_time

                    # Close connection after 5 minutes of inactivity to save resources
                    MAX_IDLE_TIME = 300  # 5 minutes
                    if idle_time > MAX_IDLE_TIME:
                        logger.bind(tag=TAG).warning(
                            f"OpenAI connection idle for {idle_time:.1f}s (max: {MAX_IDLE_TIME}s) - closing connection"
                        )
                        break

                    # Log warning after 60 seconds of inactivity
                    if idle_time > 60:
                        logger.bind(tag=TAG).debug(f"OpenAI connection idle for {idle_time:.1f}s")

                    # Send ping to keep connection alive
                    try:
                        await self.ws.ping()
                        logger.bind(tag=TAG).debug("Sent keepalive ping to OpenAI")
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
        """Handle events from OpenAI Realtime API

        Key events:
        - session.created: Session established
        - session.updated: Session configuration confirmed
        - conversation.item.created: New conversation item added
        - response.audio.delta: Streaming audio chunk
        - response.audio.done: Audio response complete
        - response.function_call_arguments.delta: Function call in progress
        - response.function_call_arguments.done: Function call complete
        - response.done: Response complete
        - error: Error occurred
        """
        # Update activity timestamp (both local and connection's)
        import time
        self.last_activity_time = time.time()
        self.conn.last_activity_time = self.last_activity_time * 1000  # Convert to milliseconds

        # Check if client has pressed abort button
        if hasattr(self.conn, 'client_abort') and self.conn.client_abort:
            logger.bind(tag=TAG).info("Client abort detected - cancelling and resetting")
            await self._cancel_response()
            # Reset abort flag immediately after first cancellation
            self.conn.client_abort = False
            return  # Skip processing this event

        event_type = event.get("type")

        # Session events
        if event_type == "session.created":
            logger.bind(tag=TAG).info("Session created successfully")

        elif event_type == "session.updated":
            session_data = event.get("session", {})
            turn_det = session_data.get("turn_detection", {})
            logger.bind(tag=TAG).info(
                f"Session configured | interrupt_response: {turn_det.get('interrupt_response')} | "
                f"create_response: {turn_det.get('create_response')} | threshold: {turn_det.get('threshold')}"
            )

        # Input audio events
        elif event_type == "input_audio_buffer.speech_started":
            import time
            self.last_speech_started_time = time.time()
            logger.bind(tag=TAG).debug("User speech detected")
            # With interrupt_response: True, OpenAI handles interruptions automatically
            # No need for manual cancellation logic

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.bind(tag=TAG).debug("Speech stopped")

        elif event_type == "input_audio_buffer.committed":
            logger.bind(tag=TAG).debug("Audio committed")

        # Conversation events
        elif event_type == "conversation.item.created":
            item = event.get("item", {})
            logger.bind(tag=TAG).debug(f"Conversation item created: {item.get('type')}")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            logger.bind(tag=TAG).info(f"User said: {transcript}")

            # Save to chat history with comprehensive error handling
            try:
                logger.bind(tag=TAG).debug(f"Attempting to save user message to dialogue - Memory enabled: {self.conn.memory is not None}")

                # Add to dialogue history (for memory)
                if hasattr(self.conn, 'dialogue') and self.conn.dialogue:
                    self.conn.dialogue.put(Message(role="user", content=transcript))
                    logger.bind(tag=TAG).debug("Successfully saved user message to dialogue")
                else:
                    logger.bind(tag=TAG).warning("dialogue object not available")

            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to save user message to dialogue: {e}", exc_info=True)

            # Report to chat history (for web UI) - runs in background with audio
            try:
                if self.conn.chat_history_conf > 0:
                    # Pass a copy of the audio buffer to avoid modification during processing
                    audio_snapshot = self.user_audio_buffer.copy()
                    enqueue_asr_report(self.conn, transcript, audio_snapshot)
                    # Clear user audio buffer after reporting
                    self.user_audio_buffer.clear()
                    logger.bind(tag=TAG).debug("Enqueued ASR report for web UI with audio")
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to enqueue ASR report: {e}", exc_info=True)

        # Response events
        elif event_type == "response.created":
            import time
            self.response_start_time = time.time()  # Track when response started
            self._emotion_sent_for_current_response = False  # Reset emotion flag for new response

            logger.bind(tag=TAG).info("AI response started")
            self.response_in_progress = True
            # Full-duplex mode: Do NOT block audio input while bot speaks
            # Audio flows in both directions simultaneously
            # Clear audio buffer for new response
            self.pcm_buffer = bytearray()
            # Reset audio sequence for new response
            self.conn._audio_sequence = 0
            self.audio_frames_sent = 0  # Track frames sent for debugging
            # Send TTS start signals - ESP32 needs them to prevent connection timeout
            # ESP32 firmware in realtime mode should keep mic active despite these signals
            await self._send_tts_initial_start()
            await self._send_tts_start()

        elif event_type == "response.output_item.added":
            item = event.get("item", {})
            logger.bind(tag=TAG).debug(f"Output item added: {item.get('type')}")

        elif event_type == "response.content_part.added":
            part = event.get("part", {})
            logger.bind(tag=TAG).debug(f"Content part added: {part.get('type')}")

        # Audio streaming
        elif event_type == "response.audio.delta":
            logger.bind(tag=TAG).debug("Received audio delta from OpenAI")
            await self._handle_audio_delta(event)

        elif event_type == "response.audio.done":
            logger.bind(tag=TAG).info(f"Audio stream complete from OpenAI - sent {self.audio_frames_sent} frames")
            # Flush any remaining audio in buffer
            await self._flush_audio_buffer()
            # Send stop signal - ESP32 needs it to prevent connection timeout
            # ESP32 firmware in realtime mode should keep connection alive and mic active
            await self._send_audio_complete()
            logger.bind(tag=TAG).info("Sent TTS stop signal")

        # Audio transcription (what the bot is saying)
        elif event_type == "response.audio_transcript.delta":
            # Don't log deltas - too verbose
            pass

        elif event_type == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            logger.bind(tag=TAG).info(f"Bot said: {transcript}")

            # Extract and send emotion/emoji to ESP32 (only once per response)
            if not hasattr(self, '_emotion_sent_for_current_response'):
                self._emotion_sent_for_current_response = False

            if not self._emotion_sent_for_current_response and transcript.strip():
                try:
                    await textUtils.get_emotion(self.conn, transcript)
                    self._emotion_sent_for_current_response = True
                    logger.bind(tag=TAG).debug("Extracted and sent emotion to ESP32")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Failed to extract emotion: {e}", exc_info=True)

            # Save to chat history with comprehensive error handling
            try:
                logger.bind(tag=TAG).debug(f"Attempting to save assistant message to dialogue - Memory enabled: {self.conn.memory is not None}")

                # Add to dialogue history (for memory)
                if hasattr(self.conn, 'dialogue') and self.conn.dialogue:
                    self.conn.dialogue.put(Message(role="assistant", content=transcript))
                    logger.bind(tag=TAG).debug("Successfully saved assistant message to dialogue")
                else:
                    logger.bind(tag=TAG).warning("dialogue object not available")

            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to save assistant message to dialogue: {e}", exc_info=True)

            # Report to chat history (for web UI) - runs in background with audio
            try:
                if self.conn.chat_history_conf > 0:
                    # Pass a copy of the audio buffer to avoid modification during processing
                    audio_snapshot = self.agent_audio_buffer.copy()
                    enqueue_tts_report(self.conn, transcript, audio_snapshot)
                    # Clear agent audio buffer after reporting
                    self.agent_audio_buffer.clear()
                    logger.bind(tag=TAG).debug("Enqueued TTS report for web UI with audio")
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to enqueue TTS report: {e}", exc_info=True)

            # Send transcription to ESP32 for display
            await self._send_transcription_to_client(transcript)

        # Text streaming (if text modality is enabled)
        elif event_type == "response.text.delta":
            delta = event.get("delta", "")
            logger.bind(tag=TAG).debug(f"Text delta: {delta}")

        elif event_type == "response.text.done":
            text = event.get("text", "")
            logger.bind(tag=TAG).debug(f"Text complete: {text}")

        # Function calling
        elif event_type == "response.function_call_arguments.delta":
            await self._handle_function_call_delta(event)

        elif event_type == "response.function_call_arguments.done":
            await self._handle_function_call_done(event)

        elif event_type == "response.output_item.done":
            item = event.get("item", {})
            item_type = item.get("type")
            logger.bind(tag=TAG).info(f"Output item done | Type: {item_type}")
            if item_type == "function_call":
                # Function call item completed, execute it
                logger.bind(tag=TAG).info(f"Function call detected: {item.get('name')}")
                await self._execute_function_call(item)
            elif item_type == "message":
                logger.bind(tag=TAG).debug(f"Message output completed")

        elif event_type == "response.done":
            # Log the full response for debugging
            response_obj = event.get("response", {})
            output_items = response_obj.get("output", [])
            logger.bind(tag=TAG).info(
                f"Response complete - sent {self.audio_frames_sent} audio frames | "
                f"Output items: {len(output_items)} | "
                f"Status: {response_obj.get('status')} | "
                f"Status details: {response_obj.get('status_details')}"
            )

            self.response_in_progress = False

            # Check if we should close connection after this response (e.g., from handle_exit_intent)
            if hasattr(self.conn, 'close_after_chat') and self.conn.close_after_chat:
                logger.bind(tag=TAG).info("Closing connection after chat as requested by handle_exit_intent")
                await self.conn.close()
            self.response_start_time = 0  # Reset response timer
            # Full-duplex mode: No blocking needed
            # Audio flows continuously - ESP32 AEC + API turn detection handles echo

        elif event_type == "response.cancelled":
            logger.bind(tag=TAG).info("Response cancelled by server")
            self.response_in_progress = False
            self.response_start_time = 0  # Reset response timer
            self.conn.client_is_speaking = False  # Bot stops speaking
            # Clear audio buffer
            self.pcm_buffer = bytearray()

        # Rate limits
        elif event_type == "rate_limits.updated":
            limits = event.get("rate_limits", [])
            logger.bind(tag=TAG).debug(f"Rate limits: {limits}")

        # Errors
        elif event_type == "error":
            error = event.get("error", {})
            error_code = error.get("code", "")

            # Handle expected errors gracefully
            if error_code == "conversation_already_has_active_response":
                # This is expected when VAD detects speech during an active response
                # Just log as warning, not error
                logger.bind(tag=TAG).warning(f"Speech detected during active response - ignoring duplicate response request")
            elif error_code == "response_cancel_not_active":
                # Trying to cancel a response that's already finished - harmless
                logger.bind(tag=TAG).debug(f"Cancel requested but response already finished - ignoring")
            else:
                # Real errors
                logger.bind(tag=TAG).error(f"Realtime API error: {error}")

        else:
            # Ignore verbose transcript events
            if event_type not in ["response.audio_transcript.delta", "response.audio_transcript.done",
                                   "response.content_part.done", "conversation.item.input_audio_transcription.delta"]:
                logger.bind(tag=TAG).debug(f"Unhandled event: {event_type}")

    async def _handle_audio_delta(self, event: Dict[str, Any]):
        """Handle streaming audio chunks from Realtime API"""
        try:
            # Get base64-encoded PCM16 audio at 24kHz
            audio_base64 = event.get("delta", "")
            if not audio_base64:
                logger.bind(tag=TAG).warning("Audio delta event has no data")
                return

            # Decode from base64
            pcm_24khz = base64.b64decode(audio_base64)

            # Downsample from 24kHz to 16kHz for ESP32
            pcm_16khz = self._resample_24khz_to_16khz(pcm_24khz)

            # Add to buffer
            self.pcm_buffer.extend(pcm_16khz)

            # Process buffer in chunks of 960 samples (1920 bytes for PCM16)
            # 960 samples = 60ms at 16kHz
            FRAME_SIZE_SAMPLES = 960
            FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2  # 2 bytes per sample (int16)

            while len(self.pcm_buffer) >= FRAME_SIZE_BYTES:
                # Check if client pressed abort button - stop sending audio immediately
                if hasattr(self.conn, 'client_abort') and self.conn.client_abort:
                    logger.bind(tag=TAG).info("Client abort detected during audio streaming - stopping")
                    self.pcm_buffer = bytearray()  # Clear buffer
                    break

                # Extract one frame
                frame_bytes = bytes(self.pcm_buffer[:FRAME_SIZE_BYTES])
                self.pcm_buffer = self.pcm_buffer[FRAME_SIZE_BYTES:]

                # Encode to Opus at 16kHz
                opus_frame = self.opus_encoder.encode(frame_bytes, FRAME_SIZE_SAMPLES)

                # Buffer Opus packet for chat history (agent audio)
                self.agent_audio_buffer.append(opus_frame)
                # Limit buffer size based on duration
                max_packets = int(self.max_audio_buffer_duration / (self.opus_frame_duration_ms / 1000))
                if len(self.agent_audio_buffer) > max_packets:
                    self.agent_audio_buffer = self.agent_audio_buffer[-max_packets:]

                # Send to client with pacing to prevent network buffer overflow
                # Each frame represents 60ms of audio, send at ~60ms intervals
                if self.conn.websocket:
                    await self.conn.send_audio_to_client(opus_frame)
                    self.audio_frames_sent += 1
                    # Pace sending: small delay to match playback rate and prevent packet drops
                    # 50ms delay = slightly faster than 60ms playback, allows some buffering
                    await asyncio.sleep(0.050)
                else:
                    logger.bind(tag=TAG).error("No websocket connection to send audio")
                    break

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling audio delta: {e}", exc_info=True)

    def _resample_24khz_to_16khz(self, pcm_24khz: bytes) -> bytes:
        """Downsample PCM audio from 24kHz to 16kHz

        Args:
            pcm_24khz: PCM16 audio at 24kHz

        Returns:
            PCM16 audio at 16kHz
        """
        import numpy as np

        # Convert bytes to int16 array
        samples_24k = np.frombuffer(pcm_24khz, dtype=np.int16)

        # Use scipy's high-quality polyphase resampling (2:3 ratio = 16kHz:24kHz)
        from scipy import signal
        samples_16k = signal.resample_poly(samples_24k, 2, 3)

        # Convert back to int16 and bytes
        return samples_16k.astype(np.int16).tobytes()

    async def _handle_function_call_delta(self, event: Dict[str, Any]):
        """Handle streaming function call arguments"""
        call_id = event.get("call_id")
        delta = event.get("delta", "")

        if call_id not in self.pending_function_calls:
            self.pending_function_calls[call_id] = {
                "name": event.get("name", ""),
                "arguments": ""
            }

        self.pending_function_calls[call_id]["arguments"] += delta

    async def _handle_function_call_done(self, event: Dict[str, Any]):
        """Handle completed function call arguments"""
        call_id = event.get("call_id")
        if call_id in self.pending_function_calls:
            logger.bind(tag=TAG).debug(f"Function call arguments complete: {call_id}")

    async def _execute_function_call(self, item: Dict[str, Any]):
        """Execute a function call using UnifiedToolHandler

        This integrates with the existing tool execution infrastructure,
        preserving all MCP, plugin, and IoT functionality.
        """
        try:
            call_id = item.get("call_id")
            function_name = item.get("name")
            arguments_str = item.get("arguments", "{}")

            logger.bind(tag=TAG).info(f"Executing function: {function_name} | Call ID: {call_id}")

            # Parse arguments
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}

            # Execute via UnifiedToolHandler
            if hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                result = await self.conn.func_handler.handle_llm_function_call(
                    self.conn,
                    {
                        "name": function_name,
                        "arguments": arguments
                    }
                )

                # Send result back to Realtime API
                await self._send_function_call_result(call_id, result, function_name)
            else:
                logger.bind(tag=TAG).error("No function handler available")
                await self._send_function_call_error(call_id, "Function handler not available")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error executing function call: {e}")
            await self._send_function_call_error(call_id, str(e))

    async def _send_function_call_result(self, call_id: str, result, function_name: str = ""):
        """Send function call result back to Realtime API"""
        try:
            # Format result for Realtime API
            output = ""
            if hasattr(result, 'response') and result.response:
                output = result.response
            elif hasattr(result, 'result') and result.result:
                output = result.result
            else:
                output = str(result)

            # Send conversation.item.create event with function call output
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output
                }
            }

            await self.ws.send(json.dumps(message))

            # For play_music, don't trigger response - music will play instead
            # Response will auto-trigger when user speaks again after music
            if function_name == "play_music":
                logger.bind(tag=TAG).info(f"Function result sent (no response trigger for music) | Call ID: {call_id}")
            else:
                # Trigger response generation for other functions
                await self.ws.send(json.dumps({"type": "response.create"}))
                logger.bind(tag=TAG).info(f"Function result sent | Call ID: {call_id}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending function result: {e}")

    async def _send_function_call_error(self, call_id: str, error: str):
        """Send function call error to Realtime API"""
        try:
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": f"Error: {error}"
                }
            }
            await self.ws.send(json.dumps(message))
            await self.ws.send(json.dumps({"type": "response.create"}))

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending function error: {e}")

    async def receive_audio(self, audio: bytes):
        """Receive audio from client and send to Realtime API

        Args:
            audio: Opus-encoded audio frame from client at 16kHz
        """
        try:
            if not self.is_connected or not self.ws:
                # Auto-connect if not yet connected
                logger.bind(tag=TAG).info("Not connected yet, auto-connecting to OpenAI Realtime...")
                await self.connect()
                if not self.is_connected or not self.ws:
                    logger.bind(tag=TAG).warning(f"Cannot receive audio - auto-connect failed")
                    return

            # Skip audio processing if music is playing
            if self.is_music_playing:
                return

            # Update activity timestamp - we're actively streaming (both local and connection's)
            import time
            self.last_activity_time = time.time()
            self.conn.last_activity_time = self.last_activity_time * 1000  # Convert to milliseconds

            # Log every 50th frame to avoid spam
            self.audio_frames_received += 1
            if self.audio_frames_received % 50 == 0:
                logger.bind(tag=TAG).info(f"Received {self.audio_frames_received} audio frames from client | Opus size: {len(audio)} bytes")

            # Log first few frames to understand Opus frame size
            if self.audio_frames_received <= 5:
                logger.bind(tag=TAG).info(f"Frame #{self.audio_frames_received}: Opus size = {len(audio)} bytes")

            # Buffer Opus packet for chat history (for voiceprint enrollment)
            self.user_audio_buffer.append(audio)
            # Limit buffer size based on duration
            max_packets = int(self.max_audio_buffer_duration / (self.opus_frame_duration_ms / 1000))
            if len(self.user_audio_buffer) > max_packets:
                self.user_audio_buffer = self.user_audio_buffer[-max_packets:]

            # Decode Opus to PCM16 at 16kHz
            # Frame size for 16kHz at 60ms = 960 samples
            try:
                pcm_16khz = self.opus_decoder.decode(audio, 960)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Opus decode error: {e} | Opus size: {len(audio)} bytes")
                return

            # Voiceprint: Buffer audio for speaker identification (if enabled and not yet identified)
            if self.voiceprint_enabled and not self.voiceprint_identified:
                await self._buffer_audio_for_voiceprint(pcm_16khz)

            # Log audio stats for debugging (every 100th frame)
            if self.audio_frames_received % 100 == 0:
                import numpy as np
                samples = np.frombuffer(pcm_16khz, dtype=np.int16)
                logger.bind(tag=TAG).info(
                    f"Audio stats - samples: {len(samples)}, "
                    f"min: {samples.min()}, max: {samples.max()}, "
                    f"mean: {samples.mean():.2f}, std: {samples.std():.2f}"
                )

            # Debug: Save received PCM audio to buffer
            if self.debug_audio_enabled:
                if self.debug_audio_start_time is None:
                    self.debug_audio_start_time = time.time()
                self.debug_pcm_buffer.extend(pcm_16khz)

                # Save to WAV file every 5 seconds of audio (16000 samples/sec * 2 bytes * 5 sec = 160000 bytes)
                if len(self.debug_pcm_buffer) >= 160000:
                    # Save in background to avoid blocking audio pipeline
                    buffer_copy = bytes(self.debug_pcm_buffer)
                    self.debug_pcm_buffer = bytearray()
                    asyncio.create_task(self._save_debug_audio(buffer_copy))

            # Upsample from 16kHz to 24kHz for Realtime API
            pcm_24khz = self._resample_16khz_to_24khz(pcm_16khz)

            # Accumulate into buffer - send in larger chunks (60ms = 1440 samples at 24kHz = 2880 bytes)
            self._input_pcm_buffer_24khz.extend(pcm_24khz)

            # Send 60ms chunks (1440 samples at 24kHz = 2880 bytes)
            chunk_size_bytes = 1440 * 2  # 2880 bytes = 60ms at 24kHz

            while len(self._input_pcm_buffer_24khz) >= chunk_size_bytes:
                chunk = bytes(self._input_pcm_buffer_24khz[:chunk_size_bytes])
                self._input_pcm_buffer_24khz = self._input_pcm_buffer_24khz[chunk_size_bytes:]

                # Encode to base64
                audio_base64 = base64.b64encode(chunk).decode('utf-8')

                # Send to Realtime API
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                }

                await self.ws.send(json.dumps(message))

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error receiving audio: {e}")

    async def _save_debug_audio(self, audio_data: bytes):
        """Save debug audio buffer to WAV file in background

        Args:
            audio_data: PCM16 audio data at 16kHz to save
        """
        try:
            import wave
            import os
            from datetime import datetime

            # Create debug directory if it doesn't exist
            debug_dir = "/tmp/audio_debug"
            os.makedirs(debug_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/openai_input_{timestamp}.wav"

            # Write WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(16000)  # 16kHz
                wf.writeframes(audio_data)

            duration = len(audio_data) / (16000 * 2)  # bytes / (samples_per_sec * bytes_per_sample)
            logger.bind(tag=TAG).info(
                f"Saved debug audio: {filename} ({duration:.2f} seconds, {len(audio_data)} bytes)"
            )

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error saving debug audio: {e}")

    def _resample_16khz_to_24khz(self, pcm_16khz: bytes) -> bytes:
        """Upsample PCM audio from 16kHz to 24kHz

        Args:
            pcm_16khz: PCM16 audio at 16kHz

        Returns:
            PCM16 audio at 24kHz
        """
        import numpy as np

        # Convert bytes to int16 array
        samples_16k = np.frombuffer(pcm_16khz, dtype=np.int16)

        # Use scipy's high-quality polyphase resampling (3:2 ratio = 24kHz:16kHz)
        from scipy import signal
        samples_24k = signal.resample_poly(samples_16k, 3, 2)

        # Convert back to int16 and bytes
        return samples_24k.astype(np.int16).tobytes()

    async def _flush_audio_buffer(self):
        """Flush any remaining audio in the buffer with padding if needed"""
        try:
            if len(self.pcm_buffer) > 0:
                logger.bind(tag=TAG).debug(f"Flushing {len(self.pcm_buffer)} bytes of remaining audio")
                # Pad to frame size if needed
                FRAME_SIZE_BYTES = 960 * 2  # 960 samples * 2 bytes
                if len(self.pcm_buffer) < FRAME_SIZE_BYTES:
                    # Pad with silence
                    padding_needed = FRAME_SIZE_BYTES - len(self.pcm_buffer)
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

    async def trigger_response(self):
        """Manually trigger response generation (public method for server-side VAD)"""
        try:
            if self.is_connected and self.ws:
                # First commit the audio buffer
                await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                # Then request response generation
                await self.ws.send(json.dumps({"type": "response.create"}))
                logger.bind(tag=TAG).info("Triggered response generation via server-side VAD")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error triggering response: {e}")

    async def cancel_response(self):
        """Cancel the current response (for interruptions)"""
        try:
            if self.is_connected and self.ws and self.response_in_progress:
                # Send response.cancel event to OpenAI
                await self.ws.send(json.dumps({"type": "response.cancel"}))
                logger.bind(tag=TAG).info("Cancelled current response due to interruption")

                # Clear audio buffer
                self.pcm_buffer = bytearray()

                # Send TTS stop signal to device
                await self._send_audio_complete()

                # Clear the input audio buffer to start fresh
                await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error cancelling response: {e}")

    async def _send_tts_initial_start(self):
        """Signal to client that audio output is starting (enables codec)"""
        try:
            if self.conn.websocket:
                await self.conn.websocket.send(
                    json.dumps({
                        "type": "tts",
                        "state": "start",
                        "session_id": self.conn.session_id
                    })
                )
                logger.bind(tag=TAG).debug("Sent TTS initial start signal to client")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS initial start: {e}")

    async def _send_tts_start(self):
        """Signal to client that TTS sentence is starting"""
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
                logger.bind(tag=TAG).debug("Sent TTS sentence_start signal to client")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS start: {e}")

    async def _send_audio_complete(self):
        """Signal to client that audio response is complete"""
        try:
            if self.conn.websocket:
                # Send TTS stop signal
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

    async def _cancel_response(self):
        """Cancel the current response (e.g., when user interrupts)"""
        try:
            if not self.ws:
                return

            # Only cancel if there's actually a response in progress
            if not self.response_in_progress:
                logger.bind(tag=TAG).debug("No response in progress to cancel")
                return

            # Send response.cancel event to OpenAI
            await self.ws.send(json.dumps({
                "type": "response.cancel"
            }))

            # Immediately mark as not in progress to prevent duplicate cancellations
            self.response_in_progress = False
            self.response_start_time = 0

            # Clear audio buffers and stop TTS playback on client
            self.pcm_buffer = bytearray()
            self._input_pcm_buffer_24khz = bytearray()
            self.debug_pcm_buffer = bytearray()
            self.debug_audio_start_time = None
            await self._send_audio_complete()

            # Don't reset client_abort here - let it persist until next response starts
            # This prevents buffered events from processing after abort

            logger.bind(tag=TAG).info("Response cancelled")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error cancelling response: {e}")

    async def _resume_audio_after_delay(self, delay_seconds: float):
        """Resume audio input after a delay (prevents echo during device playback)"""
        await asyncio.sleep(delay_seconds)
        self.conn.client_is_speaking = False
        logger.bind(tag=TAG).debug(f"Audio input resumed after {delay_seconds}s delay")

    async def _send_stop_after_delay(self, delay_seconds: float):
        """Send stop signal after a delay to allow audio playback to complete"""
        await asyncio.sleep(delay_seconds)
        await self._send_audio_complete()
        logger.bind(tag=TAG).info(f"Sent TTS stop signal after {delay_seconds:.1f}s delay")

    async def _send_transcription_to_client(self, text: str):
        """Send transcription text to client for display on ESP32"""
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
                logger.bind(tag=TAG).debug(f"Sent transcription to ESP32: {text}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending transcription: {e}")

    async def commit_audio_buffer(self):
        """Manually commit the audio buffer (for manual listening mode)"""
        try:
            if self.is_connected and self.ws:
                message = {"type": "input_audio_buffer.commit"}
                await self.ws.send(json.dumps(message))
                logger.bind(tag=TAG).debug("Audio buffer committed")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error committing audio buffer: {e}")

    async def cancel_response(self):
        """Cancel ongoing response (for user interruption)"""
        try:
            if self.is_connected and self.ws and self.response_in_progress:
                message = {"type": "response.cancel"}
                await self.ws.send(json.dumps(message))
                logger.bind(tag=TAG).info("Response cancelled")
                self.response_in_progress = False
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error cancelling response: {e}")

    async def _buffer_audio_for_voiceprint(self, pcm_data: bytes):
        """Buffer PCM audio for voiceprint identification

        Buffers audio until minimum duration is reached, then triggers identification.
        Limits buffer to maximum duration to avoid memory issues.

        Args:
            pcm_data: PCM16 audio data at 16kHz
        """
        try:
            # Add to buffer
            self.voiceprint_buffer += pcm_data

            # Calculate duration (16kHz, 16-bit = 2 bytes per sample)
            self.voiceprint_buffer_duration = len(self.voiceprint_buffer) / (16000 * 2)

            # If we've reached minimum duration and no identification task is running, start one
            if (self.voiceprint_buffer_duration >= self.voiceprint_min_duration and
                (self.voiceprint_task is None or self.voiceprint_task.done())):

                logger.bind(tag=TAG).info(
                    f"Voiceprint buffer ready ({self.voiceprint_buffer_duration:.1f}s) - starting identification"
                )

                # Start identification task in background (non-blocking)
                self.voiceprint_task = asyncio.create_task(self._identify_speaker())

            # Limit buffer size to maximum duration
            elif self.voiceprint_buffer_duration > self.voiceprint_max_duration:
                # Keep only the last max_duration seconds
                max_bytes = int(self.voiceprint_max_duration * 16000 * 2)
                self.voiceprint_buffer = self.voiceprint_buffer[-max_bytes:]
                self.voiceprint_buffer_duration = self.voiceprint_max_duration

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error buffering audio for voiceprint: {e}")

    async def _identify_speaker(self):
        """Identify speaker from buffered audio and inject into conversation context"""
        try:
            if not self.conn.voiceprint_provider:
                return

            # Convert PCM to WAV format for voiceprint API
            wav_data = self._pcm_to_wav(self.voiceprint_buffer)
            if not wav_data:
                logger.bind(tag=TAG).warning("Failed to convert PCM to WAV for voiceprint")
                return

            logger.bind(tag=TAG).info("Running voiceprint identification...")

            # Call voiceprint provider
            speaker_name = await self.conn.voiceprint_provider.identify_speaker(
                wav_data, self.conn.session_id
            )

            if speaker_name and speaker_name != "未知说话人":
                self.current_speaker = speaker_name
                self.voiceprint_identified = True

                logger.bind(tag=TAG).info(f"Speaker identified: {speaker_name}")

                # Inject speaker identity into OpenAI Realtime conversation context
                await self._inject_speaker_context(speaker_name)

                # Clear buffer after successful identification
                self.voiceprint_buffer = b""
                self.voiceprint_buffer_duration = 0.0
            else:
                logger.bind(tag=TAG).warning(
                    f"Speaker identification failed or unknown: {speaker_name}"
                )

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error identifying speaker: {e}")

    async def _inject_speaker_context(self, speaker_name: str):
        """Inject speaker identity into OpenAI Realtime conversation

        Sends a user message with speaker context to inform the LLM about speaker identity.
        This ensures the agent knows who is speaking and can personalize responses.

        Args:
            speaker_name: Identified speaker name (e.g., "Sohil", "爸爸", "妈妈")
        """
        try:
            # Add speaker context to the conversation via a user message
            # OpenAI Realtime doesn't support system messages during conversation,
            # so we inject it as a user message that the assistant will acknowledge
            context_message = f"[System Note: The current speaker has been identified as {speaker_name}. Please address them by name and personalize your responses.]"

            # Send conversation item to OpenAI Realtime
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": context_message
                        }
                    ]
                }
            }

            await self.ws.send(json.dumps(message))

            # Also add to dialogue history for memory system
            if hasattr(self.conn, 'dialogue') and self.conn.dialogue:
                from core.utils.dialogue import Message
                self.conn.dialogue.put(Message(role="system", content=context_message))

            logger.bind(tag=TAG).info(
                f"Added speaker context for {speaker_name} to OpenAI Realtime conversation"
            )

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error injecting speaker context: {e}")

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert PCM16 data to WAV format for voiceprint API

        Args:
            pcm_data: PCM16 audio data at 16kHz

        Returns:
            WAV file bytes
        """
        import io
        import wave

        if len(pcm_data) == 0:
            logger.bind(tag=TAG).warning("PCM data is empty, cannot convert to WAV")
            return b""

        # Ensure data length is even (16-bit audio)
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        try:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz sample rate
                wav_file.writeframes(pcm_data)

            wav_buffer.seek(0)
            wav_data = wav_buffer.read()

            return wav_data
        except Exception as e:
            logger.bind(tag=TAG).error(f"WAV conversion failed: {e}")
            return b""

    async def cleanup(self):
        """Clean up resources and close connections"""
        try:
            self.is_connected = False

            # Cancel voiceprint task if running
            if self.voiceprint_task and not self.voiceprint_task.done():
                self.voiceprint_task.cancel()
                try:
                    await self.voiceprint_task
                except asyncio.CancelledError:
                    pass

            if self.receive_task:
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass

            if self.ws:
                await self.ws.close()
                self.ws = None

            logger.bind(tag=TAG).info("OpenAI Realtime provider cleaned up")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during cleanup: {e}")
