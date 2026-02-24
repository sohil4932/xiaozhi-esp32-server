"""ElevenLabs Conversational AI Provider

This provider integrates ElevenLabs' Conversational AI (Agents Platform) which provides
real-time voice conversations with AI agents via WebSocket.

Architecture:
    Audio Input → ElevenLabs Agent (ASR+LLM+TTS) → Client Tool Invocation
                                           ↓
                                  Execute via UnifiedToolHandler
                                           ↓
                                  Return to ElevenLabs
                                           ↓
                                  Agent Response → Audio

Key Features:
- Ultra-low latency voice conversations (<1s)
- Natural, expressive voices with emotional intelligence
- Client tools for device control (MCP functions)
- Server tools for external API integrations
- Real-time interruption handling
- Multi-language support
- Custom agent personalities and workflows
"""

import json
import base64
import asyncio
import websockets
import opuslib_next
import numpy as np
from typing import Dict, Any, Optional
from config.logger import setup_logging
from core.handle.reportHandle import enqueue_asr_report, enqueue_tts_report
from core.utils.dialogue import Message

TAG = __name__
logger = setup_logging()


class ElevenLabsConversationalProvider:
    """ElevenLabs Conversational AI Provider

    Provides real-time voice conversations using ElevenLabs Agents Platform.
    Integrates with existing tool execution infrastructure via Client Tools.
    """

    def __init__(self, config: Dict[str, Any], conn):
        """Initialize ElevenLabs Conversational AI provider

        Args:
            config: Provider configuration containing:
                - api_key: ElevenLabs API key
                - agent_id: Conversational AI agent ID
                - use_signed_url: Whether to use signed URL (for private agents)
            conn: Connection handler instance
        """
        self.conn = conn
        self.config = config

        # ElevenLabs API configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.agent_id = config.get("agent_id")
        if not self.agent_id:
            raise ValueError("ElevenLabs agent_id is required")

        self.use_signed_url = config.get("use_signed_url", False)

        # Audio configuration
        # Input: ESP32 sends 16kHz Opus → decode → send 16kHz PCM to ElevenLabs (NO resampling)
        # Output: ElevenLabs sends 24kHz PCM → encode Opus 24kHz → send to ESP32
        #
        # IMPORTANT: ElevenLabs Conversational AI only supports 16kHz input (pcm_16000)
        # despite what conversation_initiation_metadata says. Using 16kHz directly.

        # Get actual sample rate from ESP32 for OUTPUT (24kHz from config.yaml)
        self.client_output_sample_rate = getattr(conn, 'sample_rate', 24000)  # ESP32 output: 24kHz

        # Input/Output sample rates
        self.esp32_input_sample_rate = 16000   # ESP32 sends at 16kHz
        self.elevenlabs_input_sample_rate = 16000  # ElevenLabs requires 16kHz input (pcm_16000)
        self.output_sample_rate = 24000  # ElevenLabs output: 24kHz
        self.channels = 1  # Mono
        self.opus_frame_duration_ms = 60  # 60ms Opus frames

        logger.bind(tag=TAG).info(
            f"Audio config: ESP32_input=16kHz → ElevenLabs_input=16kHz (no resampling), "
            f"ElevenLabs_output=24kHz → ESP32_output=24kHz"
        )

        # Input: Opus decode frame size (60ms at 16kHz = 960 samples)
        self.input_frame_size = 960  # 960 samples at 16kHz
        # ElevenLabs input chunk size: 960 samples = 60ms = 1920 bytes at 16kHz
        # Using 60ms chunks to match working HTML test page (instead of fragmented 10ms from ESP32)
        self.elevenlabs_chunk_samples = 960  # 60ms chunks at 16kHz
        # Output: Opus encode frame size (60ms at 24kHz = 1440 samples)
        self.output_frame_size = int(self.output_sample_rate * self.opus_frame_duration_ms / 1000)  # 1440 samples

        # PCM accumulation buffer: decode each Opus frame individually, then accumulate PCM
        # to create 60ms chunks for ElevenLabs (matching HTML test page)
        self._input_pcm_buffer = b""

        # Opus decoder at 16kHz (for ESP32 input), encoder at 24kHz (for ESP32 output)
        self.opus_decoder = opuslib_next.Decoder(self.esp32_input_sample_rate, 1)
        self.opus_encoder = opuslib_next.Encoder(self.output_sample_rate, 1, opuslib_next.APPLICATION_VOIP)

        # Buffer for partial output frames from ElevenLabs
        self._output_pcm_buffer = b""

        # WebSocket connection
        self.ws = None
        self.ws_url = None

        # Session state
        self.conversation_id = None
        self.is_connected = False
        self.is_ready = False  # True only after conversation_initiation_metadata received
        self.is_processing = False
        self.is_music_playing = False

        # Tasks
        self.receive_task = None

        # Audio state
        self.audio_session_started = False  # True after first tts start sent to client
        self.audio_frames_sent = 0
        self.audio_frames_received = 0

        # Keepalive tracking
        import time
        self.last_activity_time = time.time()

        # Client tool tracking (for MCP device functions)
        self.pending_tool_calls = {}

        # Lock to serialize all WebSocket sends (prevent concurrent send corruption)
        self._ws_send_lock = asyncio.Lock()

        logger.bind(tag=TAG).info(
            f"ElevenLabs Conversational AI provider initialized | Agent ID: {self.agent_id}"
        )

    async def connect(self):
        """Establish WebSocket connection to ElevenLabs Conversational AI"""
        try:
            # Build WebSocket URL
            if self.use_signed_url:
                # For private agents, get signed URL from API
                self.ws_url = await self._get_signed_url()
            else:
                # For public agents, use direct agent_id
                self.ws_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={self.agent_id}"

            logger.bind(tag=TAG).info(f"Connecting to ElevenLabs Conversational AI")

            # Connect to ElevenLabs using WebSocket subprotocols (per official SDK)
            # SDK uses protocols=["convai"] for public agents
            # For private agents using signed URL, auth is embedded in URL - no extra protocol needed
            # For public agents using api_key, add "bearer.<key>" subprotocol
            if self.use_signed_url:
                # Signed URL already has auth embedded - just use "convai"
                protocols = ["convai"]
                logger.bind(tag=TAG).info("Connecting with signed URL + convai subprotocol")
            else:
                # Public agent: auth via bearer subprotocol
                protocols = ["convai", f"bearer.{self.api_key}"]
                logger.bind(tag=TAG).info("Connecting with agent_id + bearer subprotocol")

            self.ws = await websockets.connect(
                self.ws_url,
                subprotocols=protocols,
                max_size=16 * 1024 * 1024,
                ping_interval=None,  # disable websockets library ping (ElevenLabs handles keepalive)
                close_timeout=10,
            )

            self.is_connected = True
            logger.bind(tag=TAG).info("Connected to ElevenLabs Conversational AI")

            # Send conversation initiation message (required by ElevenLabs protocol)
            # Must be sent before any audio; ElevenLabs responds with conversation_initiation_metadata
            initiation_msg = {
                "type": "conversation_initiation_client_data",
                "conversation_config_override": {
                    "agent": {
                        "language": "en"
                    }
                }
            }
            initiation_json = json.dumps(initiation_msg)
            logger.bind(tag=TAG).info(f"Sending conversation_initiation_client_data: {initiation_json}")
            await self.ws.send(initiation_json)
            logger.bind(tag=TAG).info("Sent conversation_initiation_client_data - waiting for conversation_initiation_metadata...")

            # Start receive task - ElevenLabs will respond with conversation_initiation_metadata
            self.receive_task = asyncio.create_task(self._receive_loop())

            # Wait for conversation_initiation_metadata before accepting audio
            # (ElevenLabs must confirm session before audio can be sent)
            for _ in range(50):  # max 5 seconds
                if self.is_ready:
                    break
                await asyncio.sleep(0.1)

            if not self.is_ready:
                logger.bind(tag=TAG).warning("Timed out waiting for conversation_initiation_metadata - proceeding anyway")
            else:
                logger.bind(tag=TAG).info("ElevenLabs session ready - audio streaming enabled")

            return True

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to connect to ElevenLabs: {e}")
            self.is_connected = False
            return False

    async def _get_signed_url(self) -> str:
        """Get signed URL for private agents

        Returns:
            Signed WebSocket URL
        """
        import aiohttp

        try:
            url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={self.agent_id}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"xi-api-key": self.api_key}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        signed_url = data.get("signed_url")
                        logger.bind(tag=TAG).info("Got signed URL for private agent")
                        return signed_url
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get signed URL: {response.status} - {error_text}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error getting signed URL: {e}")
            raise

    async def _receive_loop(self):
        """Main loop for receiving events from ElevenLabs"""
        import time

        logger.bind(tag=TAG).info("ElevenLabs receive loop started")
        messages_received = 0

        try:
            while self.is_connected and self.ws:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    messages_received += 1

                    # ElevenLabs sends all messages as JSON text
                    if isinstance(message, str):
                        event = json.loads(message)
                        # Log raw event for debugging (except ping to reduce noise)
                        if event.get("type") != "ping":
                            logger.bind(tag=TAG).debug(f"Raw event received: {message[:500]}")
                        await self._handle_event(event)
                    else:
                        # Binary messages not expected from ElevenLabs Conversational AI
                        logger.bind(tag=TAG).warning(f"Received unexpected binary message: {len(message)} bytes")

                except asyncio.TimeoutError:
                    # No message received for 30s - check idle timeout
                    idle_time = time.time() - self.last_activity_time
                    MAX_IDLE_TIME = 300  # 5 minutes
                    if idle_time > MAX_IDLE_TIME:
                        logger.bind(tag=TAG).warning(
                            f"ElevenLabs connection idle for {idle_time:.1f}s - closing"
                        )
                        break
                    # ElevenLabs handles keepalive via its own ping/pong - just continue
                    logger.bind(tag=TAG).debug(f"No message for 30s, idle_time={idle_time:.1f}s, continuing...")
                    continue

                except websockets.exceptions.ConnectionClosed as e:
                    logger.bind(tag=TAG).warning(f"WebSocket connection closed: {e.code} - {e.reason}")
                    break

                except Exception as e:
                    logger.bind(tag=TAG).error(f"Error processing message in receive loop: {e}", exc_info=True)
                    continue  # Don't break the loop on message processing errors

        except Exception as e:
            logger.bind(tag=TAG).error(f"Fatal error in receive loop: {e}", exc_info=True)
        finally:
            logger.bind(tag=TAG).info(f"ElevenLabs receive loop exited (received {messages_received} messages total)")
            self.is_connected = False
            self.is_ready = False  # Reset so next connect() waits for handshake again
            self._input_pcm_buffer = b""  # Discard any buffered input on disconnect

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle JSON events from ElevenLabs

        Event types:
        - conversation_initiation_metadata: Initial metadata when conversation starts
        - user_transcript: User's transcribed speech
        - agent_response: Agent's text response
        - agent_response_correction: Correction to previous response
        - interruption: User interrupted the agent
        - ping: Keepalive ping
        - client_tool_call: Request to execute a client tool
        """
        # Update activity timestamp
        import time
        self.last_activity_time = time.time()
        self.conn.last_activity_time = self.last_activity_time * 1000

        event_type = event.get("type")

        # Log event type only (full event details excluded for ping and audio to reduce noise)
        if event_type not in ("ping", "audio"):
            logger.bind(tag=TAG).info(f"Received ElevenLabs event: {event_type} | Full event: {event}")
        else:
            logger.bind(tag=TAG).debug(f"Received ElevenLabs event: {event_type}")

        if event_type == "conversation_initiation_metadata":
            await self._handle_conversation_init(event)

        elif event_type == "user_transcript":
            await self._handle_user_transcript(event)

        elif event_type == "agent_response":
            await self._handle_agent_response(event)

        elif event_type == "agent_response_correction":
            await self._handle_agent_correction(event)

        elif event_type == "audio":
            await self._handle_audio_event(event)

        elif event_type == "interruption":
            await self._handle_interruption(event)

        elif event_type == "client_tool_call":
            await self._handle_client_tool_call(event)

        elif event_type == "ping":
            # Per SDK: event_id is nested under ping_event.event_id
            ping_event = event.get("ping_event", {})
            await self._send_pong(ping_event.get("event_id"))

        else:
            logger.bind(tag=TAG).info(f"Unhandled ElevenLabs event: {event_type} | keys: {list(event.keys())}")

    async def _handle_conversation_init(self, event: Dict[str, Any]):
        """Handle conversation initialization metadata"""
        # Log full initialization event for debugging
        logger.bind(tag=TAG).info(f"Conversation init FULL EVENT: {event}")

        # The metadata is nested under conversation_initiation_metadata_event
        metadata = event.get("conversation_initiation_metadata_event", event)
        self.conversation_id = metadata.get("conversation_id") or event.get("conversation_id")
        agent_output_format = metadata.get("agent_output_audio_format", "unknown")
        agent_input_format = metadata.get("user_input_audio_format", "unknown")  # Correct field name

        # Extract actual sample rate from format (e.g., "pcm_16000" → 16000)
        if agent_output_format.startswith("pcm_"):
            try:
                self.output_sample_rate = int(agent_output_format.split("_")[1])
                # Recalculate output frame size based on actual sample rate
                self.output_frame_size = int(self.output_sample_rate * self.opus_frame_duration_ms / 1000)
                # Reinitialize Opus encoder with correct sample rate
                self.opus_encoder = opuslib_next.Encoder(self.output_sample_rate, 1, opuslib_next.APPLICATION_VOIP)
                logger.bind(tag=TAG).info(
                    f"Updated output config from agent metadata: {self.output_sample_rate}Hz, "
                    f"frame_size={self.output_frame_size} samples"
                )
            except (IndexError, ValueError) as e:
                logger.bind(tag=TAG).warning(f"Could not parse output format '{agent_output_format}': {e}")

        # Validate input audio format matches what we're sending
        if agent_input_format != "unknown":
            expected_format = "pcm_16000"
            if agent_input_format != expected_format:
                logger.bind(tag=TAG).warning(
                    f"⚠️ MISMATCH: Agent expects input format '{agent_input_format}' but we're sending '{expected_format}'. "
                    f"This may prevent transcription! Please reconfigure your ElevenLabs agent to accept {expected_format} input."
                )
            else:
                logger.bind(tag=TAG).info(f"✓ Input format matches: {agent_input_format}")

        logger.bind(tag=TAG).info(
            f"Conversation started | ID: {self.conversation_id} | "
            f"Input format: {agent_input_format} | Output format: {agent_output_format}"
        )
        # Signal that ElevenLabs handshake is complete - now safe to send audio
        self.is_ready = True

    async def _handle_user_transcript(self, event: Dict[str, Any]):
        """Handle user's transcribed speech"""
        try:
            # Per SDK: transcript is nested under user_transcription_event.user_transcript
            transcript_event = event.get("user_transcription_event", {})
            transcript = transcript_event.get("user_transcript", "")

            if transcript:
                logger.bind(tag=TAG).info(f"User said: {transcript}")

                # Save to dialogue for memory system
                if hasattr(self.conn, 'dialogue') and self.conn.dialogue:
                    self.conn.dialogue.put(Message(role="user", content=transcript))

                # Send transcription to client for display
                await self._send_transcription_to_client(transcript)

                # Report to management API chat history
                enqueue_asr_report(self.conn, transcript, None)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling user transcript: {e}")

    async def _handle_agent_response(self, event: Dict[str, Any]):
        """Handle agent's text response"""
        try:
            # Per SDK: response is nested under agent_response_event.agent_response
            response_event = event.get("agent_response_event", {})
            response = response_event.get("agent_response", "")

            if response:
                logger.bind(tag=TAG).info(f"Agent said: {response}")

                # Save to dialogue for memory system
                if hasattr(self.conn, 'dialogue') and self.conn.dialogue:
                    self.conn.dialogue.put(Message(role="assistant", content=response))

                # Report to management API chat history
                enqueue_tts_report(self.conn, response, None)

                # Flush any partially-accumulated PCM from the output buffer
                await self._send_audio_complete()

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling agent response: {e}")

    async def _handle_agent_correction(self, event: Dict[str, Any]):
        """Handle agent response correction"""
        try:
            correction = event.get("agent_response_correction", "")
            logger.bind(tag=TAG).info(f"Agent correction: {correction}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling agent correction: {e}")

    async def _handle_interruption(self, event: Dict[str, Any]):
        """Handle user interruption — ElevenLabs stops generating, we clear our PCM buffer"""
        try:
            logger.bind(tag=TAG).info("User interruption detected")
            # Discard any partially accumulated PCM
            self._output_pcm_buffer = b""
            logger.bind(tag=TAG).debug("PCM output buffer cleared on interruption")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling interruption: {e}")

    async def _handle_client_tool_call(self, event: Dict[str, Any]):
        """Handle client tool call request from ElevenLabs

        Event format:
        {
            "type": "client_tool_call",
            "tool_call_id": "call_xyz123",
            "tool_name": "self_audio_speaker_set_volume",
            "parameters": {"volume": 75}
        }
        """
        try:
            tool_call_id = event.get("tool_call_id")
            tool_name = event.get("tool_name")
            parameters = event.get("parameters", {})

            logger.bind(tag=TAG).info(f"Client tool call: {tool_name} | ID: {tool_call_id}")

            # Execute function via UnifiedToolHandler
            if hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                result = await self.conn.func_handler.handle_llm_function_call(
                    self.conn,
                    {
                        "name": tool_name,
                        "arguments": parameters
                    }
                )

                # Send result back to ElevenLabs
                await self._send_client_tool_result(tool_call_id, result, tool_name)
            else:
                logger.bind(tag=TAG).error("No function handler available")
                await self._send_client_tool_error(tool_call_id, "Function handler not available")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling client tool call: {e}")
            await self._send_client_tool_error(tool_call_id, str(e))

    async def _send_client_tool_result(self, tool_call_id: str, result, tool_name: str):
        """Send client tool result back to ElevenLabs"""
        try:
            # Format result
            output = ""
            if hasattr(result, 'response') and result.response:
                output = result.response
            elif hasattr(result, 'result') and result.result:
                output = result.result
            else:
                output = str(result)

            # Send tool result
            message = {
                "type": "client_tool_result",
                "tool_call_id": tool_call_id,
                "output": output,
                "is_error": False
            }

            await self._ws_send(json.dumps(message))
            logger.bind(tag=TAG).info(f"Client tool result sent | ID: {tool_call_id} | Tool: {tool_name}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending client tool result: {e}")

    async def _send_client_tool_error(self, tool_call_id: str, error: str):
        """Send client tool error to ElevenLabs"""
        try:
            message = {
                "type": "client_tool_result",
                "tool_call_id": tool_call_id,
                "output": f"Error: {error}",
                "is_error": True
            }

            await self._ws_send(json.dumps(message))
            logger.bind(tag=TAG).error(f"Client tool error sent | ID: {tool_call_id} | Error: {error}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending client tool error: {e}")

    async def _handle_audio_event(self, event: Dict[str, Any]):
        """Handle audio event from ElevenLabs (base64 PCM16 in JSON)

        Event format (per SDK):
        {
            "type": "audio",
            "audio_event": {
                "audio_base_64": "<base64-encoded PCM16 audio>",
                "event_id": 1
            }
        }
        """
        try:
            # Per SDK: audio is nested under audio_event.audio_base_64
            audio_event = event.get("audio_event", {})
            audio_b64 = audio_event.get("audio_base_64")

            if not audio_b64:
                # isFinal=True or end-of-response signal — flush remaining PCM
                is_final = event.get("isFinal", False)
                if is_final:
                    await self._send_audio_complete()
                return

            pcm_data = base64.b64decode(audio_b64)
            await self._handle_audio_output(pcm_data)

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling audio event: {e}")

    async def _handle_audio_output(self, audio_data: bytes):
        """Handle PCM16 audio output from ElevenLabs — real-time full-duplex streaming.

        Frames are sent to the client AS SOON as they arrive from ElevenLabs.
        No artificial pacing — ElevenLabs streams audio in real-time during conversation.
        The browser's streamingContext handles buffering and smooth playback.

        Args:
            audio_data: PCM16LE audio data at 24kHz from ElevenLabs
        """
        try:
            # Send tts start once per session (not per utterance) to init ESP32 decoder
            if not self.audio_session_started:
                self.audio_session_started = True
                self.conn._audio_sequence = 0
                self.audio_frames_sent = 0
                await self._send_tts_initial_start()
                await self._send_tts_start()
                logger.bind(tag=TAG).info("Agent audio session started (output: 24kHz)")

            # ElevenLabs sends 24kHz PCM - use directly, no resampling!
            # Accumulate PCM and send as 60ms Opus frames at 24kHz
            self._output_pcm_buffer += audio_data
            bytes_per_frame = self.output_frame_size * 2  # 2880 bytes at 24kHz (60ms * 24000Hz * 2 bytes)

            while len(self._output_pcm_buffer) >= bytes_per_frame:
                frame_bytes = self._output_pcm_buffer[:bytes_per_frame]
                self._output_pcm_buffer = self._output_pcm_buffer[bytes_per_frame:]

                try:
                    opus_frame = self.opus_encoder.encode(frame_bytes, self.output_frame_size)
                    if self.conn.websocket:
                        await self._do_send_opus(opus_frame)
                        self.audio_frames_sent += 1
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Opus encoding/send failed: {e}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error handling audio output: {e}")

    async def _do_send_opus(self, opus_packet: bytes):
        """Send an Opus packet to ESP32 using the correct format for the connection type.

        - Direct WebSocket (most cases): raw Opus bytes
        - MQTT gateway: 16-byte header + Opus bytes (via send_audio_to_client)
        """
        if getattr(self.conn, 'conn_from_mqtt_gateway', False):
            await self.conn.send_audio_to_client(opus_packet)
        else:
            await self.conn.websocket.send(opus_packet)

    async def receive_audio(self, audio: bytes):
        """Receive audio from ESP32 and send to ElevenLabs

        Args:
            audio: Opus-encoded audio frame from ESP32 at 16kHz
        """
        try:
            if not self.is_connected or not self.ws:
                # Auto-connect if not yet connected
                logger.bind(tag=TAG).info("Not connected yet, auto-connecting to ElevenLabs...")
                success = await self.connect()
                if not success:
                    logger.bind(tag=TAG).warning("Cannot receive audio - auto-connect failed")
                    return

            # Skip if not ready (waiting for conversation_initiation_metadata)
            if not self.is_ready:
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

            # ESP32 sends small Opus frames (10ms each). Each Opus frame must be decoded separately
            # (can't concatenate Opus frames - each has its own header). Instead, we decode each
            # frame individually and accumulate the PCM output to create 60ms chunks.

            # Decode this Opus frame to PCM16 at 16kHz
            try:
                pcm_16khz = self.opus_decoder.decode(audio, self.input_frame_size)
            except Exception as e:
                logger.bind(tag=TAG).error(f"Opus decode error: {e} | opus_len={len(audio)}B")
                return

            if len(pcm_16khz) == 0:
                logger.bind(tag=TAG).warning(f"Opus decoder returned empty PCM | opus_len={len(audio)}B")
                return

            # Log decoded PCM stats on first 5 frames
            if self.audio_frames_received <= 5:
                import numpy as np
                samples = np.frombuffer(pcm_16khz, dtype=np.int16)
                logger.bind(tag=TAG).info(
                    f"Frame #{self.audio_frames_received}: opus_in={len(audio)}B → pcm_out={len(pcm_16khz)}B ({len(samples)} samples), "
                    f"audio_range=[{samples.min()}, {samples.max()}]"
                )

            # Accumulate PCM into buffer, send in 60ms (960 sample = 1920 byte) chunks at 16kHz
            # This matches the working HTML test page frame size
            self._input_pcm_buffer += pcm_16khz
            chunk_bytes = self.elevenlabs_chunk_samples * 2  # 1920 bytes per 60ms chunk at 16kHz

            # Log pipeline info on first chunk sent
            if not hasattr(self, '_first_chunk_logged'):
                self._first_chunk_logged = False

            if not self._first_chunk_logged and len(self._input_pcm_buffer) >= chunk_bytes:
                logger.bind(tag=TAG).info(
                    f"Input audio pipeline: decode each 10ms Opus frame → accumulate PCM → "
                    f"send {chunk_bytes}B (60ms) chunks to ElevenLabs"
                )
                self._first_chunk_logged = True

            # Initialize chunk counter if not exists
            if not hasattr(self, '_chunks_sent_to_elevenlabs'):
                self._chunks_sent_to_elevenlabs = 0

            while len(self._input_pcm_buffer) >= chunk_bytes:
                chunk = self._input_pcm_buffer[:chunk_bytes]
                self._input_pcm_buffer = self._input_pcm_buffer[chunk_bytes:]

                # Log audio stats for first few chunks
                if self._chunks_sent_to_elevenlabs < 3:
                    import numpy as np
                    samples = np.frombuffer(chunk, dtype=np.int16)
                    logger.bind(tag=TAG).info(
                        f"Chunk #{self._chunks_sent_to_elevenlabs + 1}: {len(chunk)}B ({len(samples)} samples), "
                        f"range=[{samples.min()}, {samples.max()}], mean={samples.mean():.1f}, "
                        f"std={samples.std():.1f}, rms={np.sqrt(np.mean(samples.astype(np.float32)**2)):.1f}"
                    )

                audio_b64 = base64.b64encode(chunk).decode("utf-8")

                # Send audio chunk to ElevenLabs Conversational AI
                # Using same PCM16 base64 format that works for ElevenLabs STT
                try:
                    message = {
                        "user_audio_chunk": audio_b64
                    }
                    await self._ws_send(json.dumps(message))
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Failed to send audio chunk to ElevenLabs: {e}")
                    return

                self._chunks_sent_to_elevenlabs += 1
                if self._chunks_sent_to_elevenlabs == 1:
                    # Log first message details for debugging
                    import numpy as np
                    first_samples = np.frombuffer(chunk[:40], dtype=np.int16)  # First 20 samples
                    logger.bind(tag=TAG).info(
                        f"Sent first audio chunk to ElevenLabs - waiting for response... "
                        f"(chunk_size={len(chunk)}B, base64_len={len(audio_b64)} chars)"
                    )
                    logger.bind(tag=TAG).info(
                        f"First 20 PCM samples: {first_samples.tolist()}"
                    )
                    logger.bind(tag=TAG).info(
                        f"First message JSON preview: {{\"user_audio_chunk\": \"{audio_b64[:50]}...\"}} "
                        f"(showing first 50 chars of {len(audio_b64)} total)"
                    )
                elif self._chunks_sent_to_elevenlabs % 10 == 0:
                    logger.bind(tag=TAG).info(
                        f"Sent {self._chunks_sent_to_elevenlabs} audio chunks to ElevenLabs "
                        f"({self._chunks_sent_to_elevenlabs * 60}ms of audio)"
                    )

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error receiving audio: {e}")

    def _resample_16khz_to_24khz(self, pcm_16khz: bytes) -> bytes:
        """Upsample PCM audio from 16kHz to 24kHz

        This resampling smooths over frame fragmentation from ESP32 and matches
        OpenAI Realtime's successful approach.

        Args:
            pcm_16khz: PCM16 audio at 16kHz

        Returns:
            PCM16 audio at 24kHz
        """
        import numpy as np
        from scipy import signal

        # Convert bytes to int16 array
        samples_16k = np.frombuffer(pcm_16khz, dtype=np.int16)

        # Use scipy's high-quality polyphase resampling (3:2 ratio = 24kHz:16kHz)
        samples_24k = signal.resample_poly(samples_16k, 3, 2)

        # Convert back to int16 and bytes
        return samples_24k.astype(np.int16).tobytes()

    async def _ws_send(self, message: str):
        """Thread-safe WebSocket send - serializes all sends via lock"""
        async with self._ws_send_lock:
            try:
                await self.ws.send(message)
            except Exception as e:
                logger.bind(tag=TAG).error(f"WebSocket send failed: {e}")
                raise

    async def _send_pong(self, event_id: str):
        """Send pong response to ping"""
        try:
            pong_msg = {
                "type": "pong",
                "event_id": event_id
            }
            await self._ws_send(json.dumps(pong_msg))
            logger.bind(tag=TAG).debug("Sent pong to ElevenLabs")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending pong: {e}")

    async def _send_tts_initial_start(self):
        """Send TTS initial start signal to ESP32 (enables codec)"""
        try:
            if self.conn.websocket:
                tts_msg = {
                    "type": "tts",
                    "state": "start",
                    "session_id": self.conn.session_id
                }
                await self.conn.websocket.send(json.dumps(tts_msg))
                logger.bind(tag=TAG).debug("Sent TTS initial start signal")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS initial start: {e}")

    async def _send_tts_start(self):
        """Send TTS sentence start signal to ESP32"""
        try:
            if self.conn.websocket:
                tts_msg = {
                    "type": "tts",
                    "state": "sentence_start",
                    "text": "",
                    "session_id": self.conn.session_id
                }
                await self.conn.websocket.send(json.dumps(tts_msg))
                logger.bind(tag=TAG).debug("Sent TTS sentence_start signal")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error sending TTS start: {e}")

    async def _send_audio_complete(self):
        """Flush any remaining PCM as a final padded Opus frame and send it immediately."""
        try:
            bytes_per_frame = self.output_frame_size * 2
            if self._output_pcm_buffer:
                pad_needed = bytes_per_frame - (len(self._output_pcm_buffer) % bytes_per_frame)
                if pad_needed < bytes_per_frame:
                    padded = self._output_pcm_buffer + b'\x00' * pad_needed
                    try:
                        opus_frame = self.opus_encoder.encode(padded[:bytes_per_frame], self.output_frame_size)
                        if self.conn.websocket:
                            await self._do_send_opus(opus_frame)
                            self.audio_frames_sent += 1
                    except Exception:
                        pass
                self._output_pcm_buffer = b""
            logger.bind(tag=TAG).debug(f"Audio flush complete — total frames sent: {self.audio_frames_sent}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in _send_audio_complete: {e}")

    async def _send_transcription_to_client(self, text: str):
        """Send transcription text to ESP32 for display"""
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

    def _resample_to_16khz(self, pcm_data: bytes) -> bytes:
        """Resample PCM audio from client sample rate to 16kHz for ElevenLabs

        Args:
            pcm_data: PCM16 audio at client sample rate (e.g., 24kHz)

        Returns:
            PCM16 audio at 16kHz
        """
        import numpy as np
        from scipy import signal

        # Convert bytes to int16 array
        samples_client = np.frombuffer(pcm_data, dtype=np.int16)

        # Resample using polyphase filter (high quality)
        # For 24kHz → 16kHz: down=3, up=2 (ratio = 2/3 = 16/24)
        # For other rates, scipy automatically calculates the ratio
        samples_16k = signal.resample_poly(
            samples_client,
            self.elevenlabs_sample_rate,  # up (16000)
            self.client_sample_rate        # down (24000)
        )

        # Convert back to int16 and bytes
        return samples_16k.astype(np.int16).tobytes()

    def _resample_from_16khz(self, pcm_data: bytes) -> bytes:
        """Resample PCM audio from 16kHz (ElevenLabs) to client sample rate

        Args:
            pcm_data: PCM16 audio at 16kHz

        Returns:
            PCM16 audio at client sample rate (e.g., 24kHz)
        """
        import numpy as np
        from scipy import signal

        # Convert bytes to int16 array
        samples_16k = np.frombuffer(pcm_data, dtype=np.int16)

        # Resample using polyphase filter (high quality)
        # For 16kHz → 24kHz: up=3, down=2 (ratio = 3/2 = 24/16)
        samples_client = signal.resample_poly(
            samples_16k,
            self.client_sample_rate,       # up (24000)
            self.elevenlabs_sample_rate    # down (16000)
        )

        # Convert back to int16 and bytes
        return samples_client.astype(np.int16).tobytes()

    async def update_tools(self):
        """Update agent with latest client tools (called when device MCP tools become available)

        Note: ElevenLabs client tools must be configured in the agent via the web console.
        This method logs available tools for informational purposes.
        """
        try:
            if hasattr(self.conn, 'func_handler') and self.conn.func_handler:
                tools = self.conn.func_handler.get_functions()
                tool_names = [t.get("function", {}).get("name") for t in tools]
                logger.bind(tag=TAG).info(
                    f"Device MCP tools registered ({len(tools)} tools): {tool_names}. "
                    f"Note: ElevenLabs agent must be configured with client tools via web console."
                )
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to log tools: {e}")

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

            logger.bind(tag=TAG).info("ElevenLabs Conversational AI provider cleaned up")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during cleanup: {e}")
