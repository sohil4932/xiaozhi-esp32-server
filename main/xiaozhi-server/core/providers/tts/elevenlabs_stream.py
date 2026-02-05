import uuid
import json
import asyncio
import queue
import traceback
import os
from asyncio import Task
import websockets
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType
from core.utils.tts import MarkdownCleaner
from core.utils import opus_encoder_utils
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    """ElevenLabs Streaming TTS Provider

    Uses ElevenLabs WebSocket API for bidirectional streaming text-to-speech.
    Provides ultra-low latency by streaming audio as it's generated.
    """

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # Set interface type to dual stream (bidirectional)
        self.interface_type = InterfaceType.DUAL_STREAM

        # ElevenLabs configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
        self.model_id = config.get("model_id", "eleven_turbo_v2_5")  # Fastest model

        # Audio settings
        self.output_format = config.get("output_format", "pcm_16000")
        self.sample_rate = 16000
        self.audio_file_type = "pcm"

        # Voice settings
        stability = config.get("stability", "0.5")
        self.stability = float(stability) if stability else 0.5

        similarity_boost = config.get("similarity_boost", "0.75")
        self.similarity_boost = float(similarity_boost) if similarity_boost else 0.75

        # WebSocket configuration
        self.ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}&output_format={self.output_format}"
        self.ws = None
        self._monitor_task = None
        self.session_id = None

        # Opus encoder
        self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=16000, channels=1, frame_size_ms=60
        )

        logger.bind(tag=TAG).info(f"ElevenLabs TTS initialized | Voice: {self.voice_id} | Model: {self.model_id}")

    def tts_text_priority_thread(self):
        """Stream text processing thread"""
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                logger.bind(tag=TAG).debug(
                    f"收到TTS任务 | {message.sentence_type.name} | {message.content_type.name} | 会话ID: {self.conn.sentence_id}"
                )

                if message.sentence_type == SentenceType.FIRST:
                    self.conn.client_abort = False

                if self.conn.client_abort:
                    logger.bind(tag=TAG).info("收到打断信息，终止TTS文本处理线程")
                    continue

                if message.sentence_type == SentenceType.FIRST:
                    # Start new session
                    try:
                        logger.bind(tag=TAG).debug("开始启动TTS会话...")
                        future = asyncio.run_coroutine_threadsafe(
                            self.start_session(),
                            loop=self.conn.loop,
                        )
                        future.result()
                        self.before_stop_play_files.clear()
                        logger.bind(tag=TAG).debug("TTS会话启动成功")
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"启动TTS会话失败: {str(e)}")
                        continue

                elif ContentType.TEXT == message.content_type:
                    if message.content_detail:
                        try:
                            logger.bind(tag=TAG).debug(f"开始发送TTS文本: {message.content_detail}")
                            future = asyncio.run_coroutine_threadsafe(
                                self.text_to_speak(message.content_detail, None),
                                loop=self.conn.loop,
                            )
                            future.result()
                            logger.bind(tag=TAG).debug("TTS文本发送成功")
                        except Exception as e:
                            logger.bind(tag=TAG).error(f"发送TTS文本失败: {str(e)}")
                            continue

                elif ContentType.FILE == message.content_type:
                    logger.bind(tag=TAG).info(f"添加音频文件到待播放列表: {message.content_file}")
                    if message.content_file and os.path.exists(message.content_file):
                        self._process_audio_file_stream(
                            message.content_file,
                            callback=lambda audio_data: self.handle_audio_file(audio_data, message.content_detail)
                        )

                if message.sentence_type == SentenceType.LAST:
                    try:
                        logger.bind(tag=TAG).debug("开始结束TTS会话...")
                        future = asyncio.run_coroutine_threadsafe(
                            self.finish_session(),
                            loop=self.conn.loop,
                        )
                        future.result()
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"结束TTS会话失败: {str(e)}")
                        continue

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理TTS文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )

    async def _ensure_connection(self):
        """Ensure WebSocket connection is available"""
        try:
            # Check if connection exists and is still open
            if self.ws:
                try:
                    # Test if connection is still alive by attempting to ping
                    await self.ws.ping()
                    logger.bind(tag=TAG).debug("使用已有WebSocket连接")
                    return self.ws
                except:
                    # Connection is dead, close it and create new one
                    try:
                        await self.ws.close()
                    except:
                        pass
                    self.ws = None

            logger.bind(tag=TAG).info(f"建立新的WebSocket连接 | URL: {self.ws_url}")

            self.ws = await websockets.connect(
                self.ws_url,
                additional_headers={"xi-api-key": self.api_key},
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
            )

            self.session_id = uuid.uuid4().hex
            logger.bind(tag=TAG).info(f"WebSocket连接建立成功 | Session: {self.session_id}")
            return self.ws

        except Exception as e:
            logger.bind(tag=TAG).error(f"建立连接失败: {str(e)}")
            self.ws = None
            raise

    async def text_to_speak(self, text, _):
        """Send text to ElevenLabs for synthesis"""
        try:
            if self.ws is None:
                logger.bind(tag=TAG).warning("WebSocket连接不存在，终止发送文本")
                return

            filtered_text = MarkdownCleaner.clean_markdown(text)
            if filtered_text:
                message = {
                    "text": filtered_text,
                    "try_trigger_generation": True
                }
                await self.ws.send(json.dumps(message))
                logger.bind(tag=TAG).debug(f"已发送文本: {filtered_text[:50]}...")

            return

        except Exception as e:
            logger.bind(tag=TAG).error(f"发送TTS文本失败: {str(e)}")
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None
            raise

    async def start_session(self):
        """Start TTS session"""
        logger.bind(tag=TAG).debug("开始TTS会话...")
        try:
            # Close previous session if exists
            if (
                self._monitor_task is not None
                and isinstance(self._monitor_task, Task)
                and not self._monitor_task.done()
            ):
                logger.bind(tag=TAG).info("检测到未完成的上个会话，关闭监听任务和连接...")
                await self.close()

            # Establish new connection
            await self._ensure_connection()

            # Start monitor task
            self._monitor_task = asyncio.create_task(self._start_monitor_tts_response())

            # Send BOS (Beginning of Stream) message
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost
                },
                "xi_api_key": self.api_key
            }

            await self.ws.send(json.dumps(bos_message))
            logger.bind(tag=TAG).debug("BOS消息已发送")

        except Exception as e:
            logger.bind(tag=TAG).error(f"启动会话失败: {str(e)}")
            await self.close()
            raise

    async def finish_session(self):
        """Finish TTS session"""
        logger.bind(tag=TAG).debug("结束TTS会话...")
        try:
            if self.ws:
                # Send EOS (End of Stream) message
                eos_message = {
                    "text": ""
                }
                await self.ws.send(json.dumps(eos_message))
                logger.bind(tag=TAG).debug("EOS消息已发送")

                # Wait for monitor task to complete
                if self._monitor_task:
                    try:
                        await asyncio.wait_for(self._monitor_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.bind(tag=TAG).warning("等待监听任务完成超时")
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"等待监听任务完成时发生错误: {str(e)}")
                    finally:
                        self._monitor_task = None

        except Exception as e:
            logger.bind(tag=TAG).error(f"结束会话失败: {str(e)}")
            await self.close()
            raise

    async def close(self):
        """Cleanup resources"""
        if self._monitor_task:
            try:
                self._monitor_task.cancel()
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.bind(tag=TAG).warning(f"关闭时取消监听任务错误: {e}")
            self._monitor_task = None

        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None

    async def _start_monitor_tts_response(self):
        """Monitor TTS responses from ElevenLabs"""
        try:
            session_finished = False
            first_audio_sent = False
            message_count = 0

            logger.bind(tag=TAG).info("✅ TTS监听任务已启动，等待ElevenLabs响应...")

            while not self.conn.stop_event.is_set():
                try:
                    msg = await self.ws.recv()
                    message_count += 1

                    if message_count <= 5:
                        msg_size = len(msg) if isinstance(msg, (str, bytes)) else 0
                        logger.bind(tag=TAG).info(f"📨 收到第 {message_count} 条消息 | 大小: {msg_size} bytes | 类型: {type(msg).__name__}")

                    # Check for client abort
                    if self.conn.client_abort:
                        logger.bind(tag=TAG).info("收到打断信息，终止监听TTS响应")
                        break

                    # ElevenLabs sends all data as JSON (not binary WebSocket frames)
                    if isinstance(msg, str):
                        try:
                            data = json.loads(msg)

                            # Log all message types for debugging
                            if not hasattr(self, '_tts_messages_logged'):
                                self._tts_messages_logged = {}
                            msg_keys = list(data.keys())
                            msg_signature = str(sorted(msg_keys))
                            if msg_signature not in self._tts_messages_logged:
                                logger.bind(tag=TAG).debug(f"收到TTS消息类型: {msg_keys} | isFinal: {data.get('isFinal', 'N/A')}")
                                self._tts_messages_logged[msg_signature] = True

                            # Check for errors
                            if "error" in data:
                                logger.bind(tag=TAG).error(f"ElevenLabs错误: {data['error']}")
                                break

                            # Handle audio data (base64 encoded in JSON)
                            if "audio" in data and data["audio"]:
                                if not first_audio_sent:
                                    logger.bind(tag=TAG).info(f"开始接收音频数据 | 格式: {self.output_format}")
                                    self.tts_audio_queue.put((SentenceType.FIRST, [], None))
                                    first_audio_sent = True

                                # Decode base64 audio data
                                import base64
                                audio_bytes = base64.b64decode(data["audio"])

                                # Log first chunk for debugging
                                if not hasattr(self, '_logged_first_chunk'):
                                    logger.bind(tag=TAG).debug(f"首个音频块: {len(audio_bytes)} bytes")
                                    self._logged_first_chunk = True

                                # PCM16 should always have even byte count (2 bytes per sample)
                                if len(audio_bytes) % 2 != 0:
                                    logger.bind(tag=TAG).error(f"⚠️ 音频数据长度不对齐: {len(audio_bytes)} bytes - PCM16 应该是偶数!")
                                    logger.bind(tag=TAG).error(f"这可能表示输出格式不是 pcm_16000。请检查配置。")
                                    # Skip this chunk - corrupt data will cause worse problems
                                    continue

                                # Convert to Opus and queue
                                if len(audio_bytes) > 0:
                                    self.opus_encoder.encode_pcm_to_opus_stream(
                                        audio_bytes,
                                        end_of_stream=False,
                                        callback=self.handle_opus
                                    )

                            # Check if audio is complete
                            # Note: isFinal can be in the same message as the last audio chunk
                            if data.get("isFinal"):
                                logger.bind(tag=TAG).debug("收到isFinal信号，等待所有音频块处理完成")
                                # Don't break immediately - continue processing any remaining messages
                                session_finished = True

                        except json.JSONDecodeError:
                            logger.bind(tag=TAG).warning("收到无效的JSON消息")
                        except Exception as e:
                            logger.bind(tag=TAG).error(f"处理音频数据失败: {e}")

                    # Handle binary messages (shouldn't happen with ElevenLabs, but keep for safety)
                    elif isinstance(msg, (bytes, bytearray)):
                        logger.bind(tag=TAG).warning("收到意外的二进制数据，ElevenLabs应该发送JSON")

                except websockets.ConnectionClosed as e:
                    logger.bind(tag=TAG).warning(f"WebSocket连接已关闭 | Code: {e.code} | Reason: {e.reason}")
                    break
                except Exception as e:
                    logger.bind(tag=TAG).error(f"处理TTS响应时出错: {e}\n{traceback.format_exc()}")
                    break

            # Send LAST signal if session finished normally
            if session_finished:
                logger.bind(tag=TAG).debug("音频生成完成，发送结束信号")
                self._process_before_stop_play_files()

            # Close connection
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None

        finally:
            self._monitor_task = None

    def to_tts(self, text: str) -> list:
        """Non-streaming TTS for testing and file saving"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            audio_data = []

            async def _generate_audio():
                # Establish WebSocket connection
                ws = await websockets.connect(
                    self.ws_url,
                    additional_headers={"xi-api-key": self.api_key}
                )

                try:
                    # Send BOS
                    bos_message = {
                        "text": " ",
                        "voice_settings": {
                            "stability": self.stability,
                            "similarity_boost": self.similarity_boost
                        }
                    }
                    await ws.send(json.dumps(bos_message))

                    # Send text
                    filtered_text = MarkdownCleaner.clean_markdown(text)
                    text_message = {
                        "text": filtered_text,
                        "try_trigger_generation": True
                    }
                    await ws.send(json.dumps(text_message))

                    # Send EOS
                    eos_message = {"text": ""}
                    await ws.send(json.dumps(eos_message))

                    # Receive audio
                    while True:
                        msg = await ws.recv()

                        if isinstance(msg, str):
                            data = json.loads(msg)
                            if data.get("isFinal"):
                                logger.bind(tag=TAG).debug("音频生成完成")
                                break
                            if "error" in data:
                                raise Exception(f"ElevenLabs错误: {data['error']}")

                        elif isinstance(msg, (bytes, bytearray)):
                            self.opus_encoder.encode_pcm_to_opus_stream(
                                msg,
                                end_of_stream=False,
                                callback=lambda opus: audio_data.append(opus)
                            )

                finally:
                    try:
                        await ws.close()
                    except:
                        pass

            loop.run_until_complete(_generate_audio())
            loop.close()

            return audio_data

        except Exception as e:
            logger.bind(tag=TAG).error(f"生成音频数据失败: {str(e)}")
            return []
