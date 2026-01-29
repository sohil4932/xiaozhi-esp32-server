import json
import uuid
import asyncio
import websockets
import opuslib_next
from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    """ElevenLabs Streaming ASR Provider

    Uses ElevenLabs Conversational AI WebSocket API for real-time speech recognition.
    Provides ultra-low latency streaming speech-to-text.
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

        # ElevenLabs configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        # Model configuration
        self.model = config.get("model", "scribe_v2_realtime")  # STT model
        self.language = config.get("language", "en")  # Default to English

        # Audio configuration
        self.sample_rate = 16000
        self.audio_format = "pcm_16000"

        # STT configuration for faster response
        # Use VAD-based automatic commits for low latency
        vad_threshold_ms = config.get("vad_threshold_ms", "500")  # 500ms silence triggers commit

        # WebSocket URL - Using dedicated STT endpoint with VAD config
        self.ws_url = (
            f"wss://api.elevenlabs.io/v1/speech-to-text/realtime"
            f"?commit_strategy=vad"
            f"&vad_silence_threshold_ms={vad_threshold_ms}"
        )

        self.output_dir = config.get("output_dir", "./audio_output")
        self.delete_audio_file = delete_audio_file
        self.session_id = None

        logger.bind(tag=TAG).info(f"ElevenLabs ASR initialized | Model: {self.model} | Language: {self.language}")

    async def open_audio_channels(self, conn):
        await super().open_audio_channels(conn)

    async def receive_audio(self, conn, audio, audio_have_voice):
        """Receive and process audio data"""
        # Debug: Log when receive_audio is called
        if not hasattr(self, '_receive_audio_called'):
            logger.bind(tag=TAG).debug(f"receive_audio 首次调用 | audio_have_voice: {audio_have_voice} | audio: {len(audio) if audio else 0} bytes")
            self._receive_audio_called = True

        # Initialize audio cache
        if not hasattr(conn, 'asr_audio_for_voiceprint'):
            conn.asr_audio_for_voiceprint = []

        # Store audio data
        if audio:
            conn.asr_audio_for_voiceprint.append(audio)

        conn.asr_audio.append(audio)
        conn.asr_audio = conn.asr_audio[-10:]

        # Start connection when voice is detected
        if audio_have_voice and not self.is_processing and not self.asr_ws:
            logger.bind(tag=TAG).info(f"检测到语音，准备启动识别 | is_processing: {self.is_processing} | asr_ws: {self.asr_ws is not None}")
            try:
                await self._start_recognition(conn)
            except Exception as e:
                logger.bind(tag=TAG).error(f"开始识别失败: {str(e)}")
                await self._cleanup()
                return
        elif audio_have_voice:
            if not hasattr(self, '_voice_detected_but_not_starting'):
                logger.bind(tag=TAG).debug(f"检测到语音但未启动 | is_processing: {self.is_processing} | asr_ws: {self.asr_ws is not None}")
                self._voice_detected_but_not_starting = True

        # Send audio to ElevenLabs
        if self.asr_ws and self.is_processing and self.server_ready:
            try:
                # Decode Opus to PCM16
                pcm_frame = self.decoder.decode(audio, 960)

                # ElevenLabs STT expects base64-encoded PCM in specific format
                import base64
                audio_base64 = base64.b64encode(pcm_frame).decode('utf-8')

                message = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": audio_base64,
                    "sample_rate": self.sample_rate
                }

                await self.asr_ws.send(json.dumps(message))

                # Log first few chunks for debugging
                if not hasattr(self, '_audio_chunks_sent'):
                    self._audio_chunks_sent = 0
                self._audio_chunks_sent += 1
                if self._audio_chunks_sent <= 3:
                    logger.bind(tag=TAG).debug(f"已发送音频块 #{self._audio_chunks_sent}, PCM大小: {len(pcm_frame)} bytes")

            except Exception as e:
                logger.bind(tag=TAG).warning(f"发送音频失败: {str(e)}")
                await self._cleanup(conn)

    async def _start_recognition(self, conn):
        """Start recognition session"""
        try:
            # Establish WebSocket connection
            logger.bind(tag=TAG).info(f"建立STT连接 | URL: {self.ws_url}")

            self.asr_ws = await websockets.connect(
                self.ws_url,
                additional_headers={"xi-api-key": self.api_key},
                max_size=1000000000,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )

            self.session_id = uuid.uuid4().hex
            logger.bind(tag=TAG).info(f"STT WebSocket连接建立成功 | Session: {self.session_id}")

            self.is_processing = True
            self.server_ready = True  # Ready to send audio immediately
            self.forward_task = asyncio.create_task(self._forward_results(conn))

            # Send cached audio if available
            if conn.asr_audio:
                logger.bind(tag=TAG).debug(f"发送缓存音频 ({len(conn.asr_audio)} 块)...")
                import base64
                for cached_audio in conn.asr_audio[-10:]:
                    try:
                        pcm_frame = self.decoder.decode(cached_audio, 960)
                        audio_base64 = base64.b64encode(pcm_frame).decode('utf-8')
                        message = {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": audio_base64,
                            "sample_rate": self.sample_rate
                        }
                        await self.asr_ws.send(json.dumps(message))
                    except Exception as e:
                        logger.bind(tag=TAG).warning(f"发送缓存音频失败: {e}")
                        break

            logger.bind(tag=TAG).debug("STT已准备接收音频")

        except Exception as e:
            logger.bind(tag=TAG).error(f"启动识别失败: {str(e)}")
            await self._cleanup()
            raise

    async def _forward_results(self, conn):
        """Forward recognition results"""
        try:
            while not conn.stop_event.is_set():
                try:
                    response = await asyncio.wait_for(self.asr_ws.recv(), timeout=1.0)

                    # Parse JSON response
                    result = json.loads(response)
                    message_type = result.get("message_type", "")

                    # Log all message types for debugging
                    if not hasattr(self, '_message_types_logged'):
                        self._message_types_logged = set()
                    if message_type not in self._message_types_logged:
                        logger.bind(tag=TAG).debug(f"收到消息类型: {message_type} | 完整消息: {result}")
                        self._message_types_logged.add(message_type)

                    # Handle different message types from STT API
                    if message_type == "session_started":
                        logger.bind(tag=TAG).debug(f"会话已启动: {result}")
                        continue

                    elif message_type == "partial_transcript":
                        # Partial transcription (not committed yet)
                        text = result.get("text", "")
                        if text:
                            logger.bind(tag=TAG).debug(f"部分识别: {text}")
                            # We'll wait for committed_transcript before processing

                    elif message_type == "committed_transcript":
                        # Final committed transcription
                        text = result.get("text", "")
                        if text:
                            logger.bind(tag=TAG).info(f"识别到文本: {text}")

                            # Handle manual vs automatic mode
                            if conn.client_listen_mode == "manual":
                                if self.text:
                                    self.text += " " + text
                                else:
                                    self.text = text

                                # Trigger processing on stop signal
                                if conn.client_voice_stop:
                                    audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                                    if len(audio_data) > 0:
                                        logger.bind(tag=TAG).debug("收到最终识别结果，触发处理")
                                        await self.handle_voice_stop(conn, audio_data)
                                        conn.asr_audio.clear()
                                        conn.reset_vad_states()
                                    break
                            else:
                                # Automatic mode - process immediately
                                self.text = text
                                conn.reset_vad_states()
                                audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                                await self.handle_voice_stop(conn, audio_data)
                                break

                    elif message_type == "committed_transcript_with_timestamps":
                        # Committed transcript with word-level timestamps
                        text = result.get("text", "")
                        if text:
                            logger.bind(tag=TAG).info(f"识别到文本（含时间戳）: {text}")
                            # Process same as committed_transcript
                            if conn.client_listen_mode == "manual":
                                if self.text:
                                    self.text += " " + text
                                else:
                                    self.text = text
                                if conn.client_voice_stop:
                                    audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                                    if len(audio_data) > 0:
                                        await self.handle_voice_stop(conn, audio_data)
                                        conn.asr_audio.clear()
                                        conn.reset_vad_states()
                                    break
                            else:
                                self.text = text
                                conn.reset_vad_states()
                                audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                                await self.handle_voice_stop(conn, audio_data)
                                break

                    elif message_type == "error":
                        error_msg = result.get("message", "Unknown error")
                        logger.bind(tag=TAG).error(f"识别错误: {error_msg}")
                        break

                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    logger.bind(tag=TAG).info("ASR服务连接已关闭")
                    self.is_processing = False
                    break
                except json.JSONDecodeError as e:
                    logger.bind(tag=TAG).warning(f"JSON解析失败: {e}")
                    continue
                except Exception as e:
                    logger.bind(tag=TAG).error(f"处理结果失败: {str(e)}")
                    break

        except Exception as e:
            logger.bind(tag=TAG).error(f"结果转发失败: {str(e)}")
        finally:
            await self._cleanup()
            if conn:
                if hasattr(conn, 'asr_audio_for_voiceprint'):
                    conn.asr_audio_for_voiceprint = []
                if hasattr(conn, 'asr_audio'):
                    conn.asr_audio = []

    async def _cleanup(self, conn=None):
        """Cleanup resources"""
        logger.bind(tag=TAG).debug(f"开始ASR会话清理 | 当前状态: processing={self.is_processing}, server_ready={self.server_ready}")

        # Reset state
        self.is_processing = False
        self.server_ready = False
        logger.bind(tag=TAG).debug("ASR状态已重置")

        # Close connection
        if self.asr_ws:
            try:
                logger.bind(tag=TAG).debug("正在关闭WebSocket连接")
                await asyncio.wait_for(self.asr_ws.close(), timeout=2.0)
                logger.bind(tag=TAG).debug("WebSocket连接已关闭")
            except Exception as e:
                logger.bind(tag=TAG).error(f"关闭WebSocket连接失败: {e}")
            finally:
                self.asr_ws = None

        # Clear task reference
        self.forward_task = None

        logger.bind(tag=TAG).debug("ASR会话清理完成")

    async def _send_stop_request(self):
        """Send stop request to commit final transcription (for manual mode)"""
        if self.asr_ws:
            try:
                # Stop sending audio
                self.is_processing = False
                logger.bind(tag=TAG).debug("停止音频发送，请求最终识别结果")

                # Send commit message to get final transcription
                commit_message = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",  # Empty audio
                    "commit": True  # Request final transcription
                }
                await self.asr_ws.send(json.dumps(commit_message))
                logger.bind(tag=TAG).debug("已发送提交请求")

            except Exception as e:
                logger.bind(tag=TAG).error(f"发送停止请求失败: {e}")

    async def speech_to_text(self, opus_data, session_id, audio_format):
        """Get recognition result"""
        result = self.text
        self.text = ""
        return result, None

    async def close(self):
        """Close resources"""
        await self._cleanup(None)
        if hasattr(self, 'decoder') and self.decoder is not None:
            try:
                del self.decoder
                self.decoder = None
                logger.bind(tag=TAG).debug("ElevenLabs decoder resources released")
            except Exception as e:
                logger.bind(tag=TAG).debug(f"释放ElevenLabs decoder资源时出错: {e}")
