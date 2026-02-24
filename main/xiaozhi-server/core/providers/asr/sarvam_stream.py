import json
import base64
import asyncio
import websockets
import opuslib_next
from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    """Sarvam AI Streaming ASR Provider (Saarika v2.5)

    Uses Sarvam AI WebSocket API for real-time speech recognition.
    Supports 10+ Indian languages with ultra-low latency.
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

        # Sarvam AI configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Sarvam AI API key is required")

        # Model configuration
        self.model = config.get("model", "saarika:v2.5")  # STT model
        self.language = config.get("language_code", "hi-IN")  # Default to Hindi

        # Audio configuration
        self.sample_rate = config.get("sample_rate", 16000)
        self.input_audio_codec = config.get("input_audio_codec", "pcm")

        # VAD configuration
        high_vad_sensitivity = config.get("high_vad_sensitivity", True)
        vad_signals = config.get("vad_signals", True)
        flush_signal = config.get("flush_signal", True)

        # Build WebSocket URL with query parameters
        self.ws_url = (
            f"wss://api.sarvam.ai/speech-to-text/ws"
            f"?language-code={self.language}"
            f"&model={self.model}"
            f"&sample_rate={self.sample_rate}"
            f"&high_vad_sensitivity={'true' if high_vad_sensitivity else 'false'}"
            f"&vad_signals={'true' if vad_signals else 'false'}"
            f"&flush_signal={'true' if flush_signal else 'false'}"
        )

        self.output_dir = config.get("output_dir", "./audio_output")
        self.delete_audio_file = delete_audio_file
        self.session_id = None

        # Track statistics
        self.empty_transcript_count = 0
        self.total_transcript_count = 0

        logger.bind(tag=TAG).info(
            f"Sarvam AI ASR initialized | Model: {self.model} | Language: {self.language} | Sample Rate: {self.sample_rate}"
        )
        logger.bind(tag=TAG).info(f"ASR WebSocket URL: {self.ws_url}")

    async def open_audio_channels(self, conn):
        await super().open_audio_channels(conn)

    async def receive_audio(self, conn, audio, audio_have_voice):
        """Receive and process audio data"""
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
            logger.bind(tag=TAG).info(f"检测到语音，准备启动识别")
            try:
                await self._start_recognition(conn)
            except Exception as e:
                logger.bind(tag=TAG).error(f"开始识别失败: {str(e)}")
                await self._cleanup()
                return

        # Send audio to Sarvam AI
        if self.asr_ws and self.is_processing and self.server_ready:
            try:
                # Decode Opus to PCM16
                pcm_data = self.decoder.decode(bytes(audio), frame_size=960)

                # Encode to base64
                audio_base64 = base64.b64encode(pcm_data).decode('utf-8')

                # Send audio message in correct format
                message = {
                    "audio": {
                        "data": audio_base64,
                        "sample_rate": str(self.sample_rate),
                        "encoding": "audio/wav"
                    }
                }

                # Log first few audio sends
                if not hasattr(self, '_audio_send_count'):
                    self._audio_send_count = 0
                self._audio_send_count += 1
                if self._audio_send_count <= 5:
                    logger.bind(tag=TAG).info(f"🎤 发送音频 #{self._audio_send_count} | PCM: {len(pcm_data)} bytes | Base64: {len(audio_base64)} chars")

                await self.asr_ws.send(json.dumps(message))

            except Exception as e:
                logger.bind(tag=TAG).error(f"发送音频数据失败: {e}")
                await self._cleanup()

    async def _start_recognition(self, conn):
        """Start speech recognition session"""
        if self.is_processing:
            logger.bind(tag=TAG).warning("识别已在进行中，跳过")
            return

        try:
            # Connect to WebSocket
            logger.bind(tag=TAG).debug(f"连接到 Sarvam AI STT: {self.ws_url}")

            self.asr_ws = await websockets.connect(
                self.ws_url,
                additional_headers={
                    "Api-Subscription-Key": self.api_key
                },
                max_size=1000000000,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )

            logger.bind(tag=TAG).info("STT WebSocket连接建立成功")

            # No initial config needed - all config in URL query parameters
            self.is_processing = True
            self.server_ready = True

            # Start listening for responses
            self.forward_task = asyncio.create_task(self._listen_for_responses(conn))

        except Exception as e:
            logger.bind(tag=TAG).error(f"启动识别会话失败: {e}")
            await self._cleanup()
            raise

    async def _listen_for_responses(self, conn):
        """Listen for transcription responses from Sarvam AI"""
        try:
            while self.is_processing:
                try:
                    # Use shorter timeout to check for user stop more frequently
                    response = await asyncio.wait_for(self.asr_ws.recv(), timeout=1.0)

                    result = json.loads(response)
                    message_type = result.get("type", "")

                    # Log ALL messages for debugging
                    logger.bind(tag=TAG).info(f"收到STT消息 | type: {message_type} | 完整: {result}")

                    # Handle different message types from Sarvam API
                    if message_type == "data":
                        # This is the actual transcript response
                        data = result.get("data", {})
                        transcript = data.get("transcript", "")

                        # Track statistics
                        self.total_transcript_count += 1

                        # Process transcripts
                        if transcript:
                            logger.bind(tag=TAG).info(f"✅ 识别到文本: {transcript}")

                            # Accumulate all transcripts
                            if self.text:
                                self.text += " " + transcript
                            else:
                                self.text = transcript
                        else:
                            # Log when we get empty transcripts
                            self.empty_transcript_count += 1
                            logger.bind(tag=TAG).warning(
                                f"⚠️ 收到空文本 ({self.empty_transcript_count}/{self.total_transcript_count}) | "
                                f"audio_duration: {data.get('metrics', {}).get('audio_duration', 0):.3f}s | "
                                f"processing_latency: {data.get('metrics', {}).get('processing_latency', 0):.3f}s"
                            )

                        # Check if user stopped (for both empty and non-empty transcripts)
                        if conn.client_listen_mode == "manual":
                            if transcript:
                                logger.bind(tag=TAG).info(f"手动模式: 累积文本 | 当前累积: {self.text}")

                            # If user clicked stop and we have accumulated text, process it
                            if conn.client_voice_stop and self.text:
                                audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                                logger.bind(tag=TAG).info(f"✅ 手动模式: 用户已停止，处理累积文本 '{self.text}'")
                                await self.handle_voice_stop(conn, audio_data)
                                conn.asr_audio.clear()
                                conn.reset_audio_states()
                                break
                        else:
                            # Auto mode: process immediately when we have text
                            if transcript:
                                conn.reset_audio_states()
                                audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                                logger.bind(tag=TAG).info("自动模式下收到识别结果，触发处理")
                                await self.handle_voice_stop(conn, audio_data)
                                break

                    elif message_type == "events":
                        # Handle event messages (START_SPEECH, END_SPEECH)
                        event_data = result.get("data", {})
                        signal_type = event_data.get("signal_type", "")

                        if signal_type == "START_SPEECH":
                            logger.bind(tag=TAG).info("🎤 检测到语音开始")

                        elif signal_type == "END_SPEECH":
                            logger.bind(tag=TAG).info("🛑 检测到语音结束")

                            # In manual mode, handle based on whether user stopped
                            if conn.client_listen_mode == "manual":
                                # Always send flush on END_SPEECH to get any remaining transcript
                                flush_message = {"type": "flush"}
                                await self.asr_ws.send(json.dumps(flush_message))

                                if conn.client_voice_stop:
                                    # User has stopped - if we have text, process it immediately
                                    if self.text:
                                        logger.bind(tag=TAG).info(f"手动模式: 用户已停止且有累积文本，立即处理: '{self.text}'")
                                        audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                                        await self.handle_voice_stop(conn, audio_data)
                                        conn.asr_audio.clear()
                                        conn.reset_audio_states()
                                        break
                                    else:
                                        logger.bind(tag=TAG).info("手动模式: 用户已停止但无文本，继续等待")
                                else:
                                    logger.bind(tag=TAG).info("手动模式: 语音结束但用户未停止，发送flush获取中间结果")

                    elif message_type == "error":
                        error_msg = result.get("message", "Unknown error")
                        logger.bind(tag=TAG).error(f"❌ Sarvam API错误: {error_msg}")
                        break

                except asyncio.TimeoutError:
                    # Timeout - check if user stopped in manual mode
                    if conn.client_listen_mode == "manual" and conn.client_voice_stop and self.text:
                        logger.bind(tag=TAG).info(f"⏱️ 超时检测: 用户已停止且有文本，立即处理: '{self.text}'")
                        audio_data = getattr(conn, 'asr_audio_for_voiceprint', [])
                        await self.handle_voice_stop(conn, audio_data)
                        conn.asr_audio.clear()
                        conn.reset_audio_states()
                        break
                    continue
                except websockets.ConnectionClosed:
                    logger.bind(tag=TAG).info("WebSocket连接已关闭")
                    break

        except Exception as e:
            logger.bind(tag=TAG).error(f"监听响应时出错: {e}")
        finally:
            await self._cleanup()
            if conn:
                if hasattr(conn, 'asr_audio_for_voiceprint'):
                    conn.asr_audio_for_voiceprint = []
                if hasattr(conn, 'asr_audio'):
                    conn.asr_audio = []

    async def _send_stop_request(self):
        """Send stop request to finalize transcription"""
        if self.asr_ws:
            try:
                # Stop sending audio
                self.is_processing = False
                logger.bind(tag=TAG).debug("停止音频发送")

                # Send flush signal to get final transcription
                flush_message = {"type": "flush"}
                await self.asr_ws.send(json.dumps(flush_message))
                logger.bind(tag=TAG).debug("已发送flush信号")

            except Exception as e:
                logger.bind(tag=TAG).error(f"发送停止请求失败: {e}")

    async def _cleanup(self):
        """Cleanup WebSocket connection and tasks"""
        self.is_processing = False
        self.server_ready = False

        if self.forward_task:
            try:
                self.forward_task.cancel()
                await self.forward_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.bind(tag=TAG).warning(f"取消监听任务时出错: {e}")
            self.forward_task = None

        if self.asr_ws:
            try:
                await self.asr_ws.close()
            except:
                pass
            self.asr_ws = None

        logger.bind(tag=TAG).debug("清理完成")

    async def speech_to_text(self, opus_data, session_id, audio_format="opus"):
        """
        Get recognition result (called by base class after handle_voice_stop completes).
        Returns accumulated text from streaming recognition.
        """
        result = self.text
        self.text = ""
        logger.bind(tag=TAG).info(f"📤 返回识别结果给base class: '{result}' | 长度: {len(result)} 字符")
        return result, None

    async def close(self):
        """Close the ASR provider"""
        await self._cleanup()
