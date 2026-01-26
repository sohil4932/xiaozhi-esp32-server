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
    """Sarvam AI Streaming TTS Provider (Bulbul v2/v3)

    Uses Sarvam AI WebSocket API for bidirectional streaming text-to-speech.
    Provides ultra-low latency with support for 22 Indian languages.
    """

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # Set interface type to dual stream (bidirectional)
        self.interface_type = InterfaceType.DUAL_STREAM

        # Sarvam AI configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Sarvam AI API key is required")

        # Model configuration
        self.model = config.get("model", "bulbul:v3")  # bulbul:v2, bulbul:v3-beta, bulbul:v3
        self.speaker = config.get("speaker", "meera")  # Voice speaker name
        self.target_language_code = config.get("target_language_code", "hi-IN")  # Default Hindi

        # Voice settings
        pitch = config.get("pitch", "0")
        self.pitch = float(pitch) if pitch else 0.0

        pace = config.get("pace", "1.0")
        self.pace = float(pace) if pace else 1.0

        loudness = config.get("loudness", "1.0")
        self.loudness = float(loudness) if loudness else 1.0

        # Buffer settings
        min_buffer_size = config.get("min_buffer_size", "50")
        self.min_buffer_size = int(min_buffer_size) if min_buffer_size else 50

        max_chunk_length = config.get("max_chunk_length", "200")
        self.max_chunk_length = int(max_chunk_length) if max_chunk_length else 200

        # Audio settings
        self.output_audio_codec = config.get("output_audio_codec", "pcm")
        self.output_audio_bitrate = config.get("output_audio_bitrate", "16000")
        self.sample_rate = 16000
        self.audio_file_type = "pcm"

        # WebSocket configuration
        send_completion_event = config.get("send_completion_event", "true")
        self.ws_url = f"wss://api.sarvam.ai/text-to-speech/ws?model={self.model}&send_completion_event={send_completion_event}"
        self.ws = None
        self._monitor_task = None
        self.session_id = None

        # Opus encoder
        self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=16000, channels=1, frame_size_ms=60
        )

        # Text buffering for smoother speech
        self.text_buffer = ""
        self.buffer_lock = asyncio.Lock()

        # Punctuation marks that trigger sending buffered text
        self.sentence_endings = ("।", ".", "?", "!", "？", "！", "।", "॥")  # Including Gujarati/Hindi punctuation
        self.phrase_breaks = (",", "，", ";", "；", ":", "：")

        logger.bind(tag=TAG).info(
            f"Sarvam AI TTS initialized | Model: {self.model} | Speaker: {self.speaker} | Language: {self.target_language_code}"
        )

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
                continue

    async def _send_buffered_text(self, force_send=False):
        """Send buffered text to TTS if conditions are met"""
        async with self.buffer_lock:
            if not self.text_buffer:
                return

            should_send = force_send

            # Check if buffer has enough content or ends with punctuation
            if not should_send:
                buffer_len = len(self.text_buffer)
                ends_with_sentence = self.text_buffer.rstrip().endswith(self.sentence_endings)
                ends_with_phrase = self.text_buffer.rstrip().endswith(self.phrase_breaks)

                # Send if: buffer is large enough, or ends with sentence/phrase punctuation
                should_send = (
                    buffer_len >= self.min_buffer_size or
                    ends_with_sentence or
                    (ends_with_phrase and buffer_len >= 10)  # Send on phrase break if we have some content
                )

            if should_send:
                text_to_send = self.text_buffer.strip()
                self.text_buffer = ""  # Clear buffer

                if text_to_send:
                    # Check if text has any actual content (not just whitespace/punctuation)
                    import re
                    has_content = bool(re.search(r'[\w\u0A80-\u0AFF\u0900-\u097F]', text_to_send))
                    if not has_content:
                        logger.bind(tag=TAG).debug(f"⚠️ 缓冲文本无有效内容，跳过 | 文本: {text_to_send}")
                        return

                    # Send text message
                    text_message = {
                        "type": "text",
                        "data": {
                            "text": text_to_send
                        }
                    }

                    await self.ws.send(json.dumps(text_message))
                    logger.bind(tag=TAG).debug(f"已发送缓冲文本 | 长度: {len(text_to_send)}")

    async def text_to_speak(self, text, output_file=None):
        """Send text to TTS service for synthesis with buffering"""
        try:
            if self.ws is None:
                logger.bind(tag=TAG).warning("WebSocket连接不存在，终止发送文本")
                return

            # Filter Markdown
            filtered_text = MarkdownCleaner.clean_markdown(text)

            if not filtered_text or not filtered_text.strip():
                return

            # Add to buffer
            async with self.buffer_lock:
                self.text_buffer += filtered_text

            # Try to send if conditions are met
            await self._send_buffered_text(force_send=False)

            return
        except Exception as e:
            logger.bind(tag=TAG).error(f"发送TTS文本失败: {str(e)}")
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None

    async def start_session(self):
        """Start TTS session"""
        logger.bind(tag=TAG).debug("初始化TTS会话...")
        try:
            # Close previous session if exists
            if (
                self._monitor_task is not None
                and isinstance(self._monitor_task, Task)
                and not self._monitor_task.done()
            ):
                logger.bind(tag=TAG).info("检测到未完成的上个会话，关闭监听任务和连接...")
                await self.close()

            # Reset text buffer for new session
            async with self.buffer_lock:
                self.text_buffer = ""
            logger.bind(tag=TAG).debug("已重置文本缓冲")

            # Connect to WebSocket
            logger.bind(tag=TAG).info(f"🔗 连接TTS WebSocket: {self.ws_url}")
            logger.bind(tag=TAG).info(f"🔑 使用API Key: {self.api_key[:10]}...")

            self.ws = await websockets.connect(
                self.ws_url,
                additional_headers={"Api-Subscription-Key": self.api_key},
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
            )

            logger.bind(tag=TAG).info("✅ TTS WebSocket连接成功")

            # Start monitoring responses BEFORE sending config
            # This ensures we don't miss any early responses
            self._monitor_task = asyncio.create_task(self._start_monitor_tts_response())

            # Send initial config message in correct format
            # Note: Sarvam is very strict about types - use integers for whole numbers
            config_data = {
                "target_language_code": self.target_language_code,
                "speaker": self.speaker,
                "pitch": int(float(self.pitch)) if float(self.pitch) == int(float(self.pitch)) else float(self.pitch),
                "pace": int(float(self.pace)) if float(self.pace) == int(float(self.pace)) else float(self.pace),
                "loudness": int(float(self.loudness)) if float(self.loudness) == int(float(self.loudness)) else float(self.loudness),
                "speech_sample_rate": str(self.output_audio_bitrate),
                "output_audio_codec": "wav"
            }

            config_message = {
                "type": "config",
                "data": config_data
            }

            await self.ws.send(json.dumps(config_message))
            logger.bind(tag=TAG).info(f"📤 已发送配置消息: {config_message}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"❌ 启动会话失败: {str(e)}")
            logger.bind(tag=TAG).error(f"错误类型: {type(e).__name__}")
            if hasattr(e, 'status_code'):
                logger.bind(tag=TAG).error(f"HTTP状态码: {e.status_code}")
            if hasattr(e, 'headers'):
                logger.bind(tag=TAG).error(f"响应头: {e.headers}")
            await self.close()
            raise

    async def finish_session(self):
        """Finish TTS session"""
        logger.bind(tag=TAG).debug("结束TTS会话...")
        try:
            if self.ws:
                # Send any remaining buffered text
                await self._send_buffered_text(force_send=True)
                logger.bind(tag=TAG).debug("已强制发送剩余缓冲文本")

                # Send flush message to finalize audio in correct format
                flush_message = {"type": "flush"}
                await self.ws.send(json.dumps(flush_message))
                logger.bind(tag=TAG).debug("已发送flush消息")

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
        """Monitor TTS responses from Sarvam AI"""
        try:
            session_finished = False
            first_audio_sent = False
            message_count = 0

            logger.bind(tag=TAG).info("✅ TTS监听任务已启动，等待Sarvam AI响应...")

            while not self.conn.stop_event.is_set():
                try:
                    msg = await self.ws.recv()
                    message_count += 1

                    if message_count <= 5:
                        msg_size = len(msg) if isinstance(msg, (str, bytes)) else 0
                        logger.bind(tag=TAG).info(f"📨 收到第 {message_count} 条消息 | 大小: {msg_size} bytes | 类型: {type(msg).__name__}")
                        # Log actual message content for debugging
                        if isinstance(msg, str) and msg_size < 500:
                            logger.bind(tag=TAG).info(f"消息内容: {msg}")

                    # Check for client abort
                    if self.conn.client_abort:
                        logger.bind(tag=TAG).info("收到打断信息，终止监听TTS响应")
                        break

                    # Sarvam AI sends audio as JSON
                    if isinstance(msg, str):
                        try:
                            data = json.loads(msg)

                            # Log message types for debugging
                            if not hasattr(self, '_tts_messages_logged'):
                                self._tts_messages_logged = {}
                            msg_keys = list(data.keys())
                            msg_signature = str(sorted(msg_keys))
                            if msg_signature not in self._tts_messages_logged:
                                logger.bind(tag=TAG).debug(f"收到TTS消息类型: {msg_keys}")
                                self._tts_messages_logged[msg_signature] = True

                            # Get message type
                            msg_type = data.get("type", "")

                            # Check for errors
                            if msg_type == "error":
                                error_info = data.get("data", {})
                                error_msg = error_info.get("message", "Unknown error")
                                error_code = error_info.get("code", "")
                                logger.bind(tag=TAG).error(f"❌ Sarvam TTS错误 [Code: {error_code}]: {error_msg}")
                                break

                            # Handle audio response in correct format
                            if msg_type == "audio":
                                audio_data = data.get("data", {})
                                audio_base64 = audio_data.get("audio", "")

                                if audio_base64:
                                    if not first_audio_sent:
                                        logger.bind(tag=TAG).info(f"开始接收音频数据 | 编码: {self.output_audio_codec}")
                                        self.tts_audio_queue.put((SentenceType.FIRST, [], None))
                                        first_audio_sent = True

                                    # Decode base64 audio data
                                    import base64
                                    audio_bytes = base64.b64decode(audio_base64)

                                    # Log first chunk for debugging
                                    if not hasattr(self, '_logged_first_chunk'):
                                        logger.bind(tag=TAG).debug(f"首个音频块: {len(audio_bytes)} bytes")
                                        self._logged_first_chunk = True

                                    # Convert to Opus and queue
                                    if len(audio_bytes) > 0:
                                        self.opus_encoder.encode_pcm_to_opus_stream(
                                            audio_bytes,
                                            end_of_stream=False,
                                            callback=self.handle_opus
                                        )

                            # Check if audio is complete (event with final type)
                            elif msg_type == "event":
                                event_data = data.get("data", {})
                                event_type = event_data.get("event_type", "")
                                if event_type == "final":
                                    logger.bind(tag=TAG).debug("收到final事件信号，音频生成完成")
                                    session_finished = True
                                    # Break immediately to send LAST signal
                                    break

                            # Check if audio is complete (completion type - alternative format)
                            elif msg_type == "completion":
                                logger.bind(tag=TAG).debug("收到completion信号，音频生成完成")
                                session_finished = True
                                break

                        except json.JSONDecodeError:
                            logger.bind(tag=TAG).warning("收到无效的JSON消息")
                        except Exception as e:
                            logger.bind(tag=TAG).error(f"处理音频数据失败: {e}")

                    # Handle binary messages (shouldn't happen with Sarvam, but keep for safety)
                    elif isinstance(msg, (bytes, bytearray)):
                        logger.bind(tag=TAG).warning("收到意外的二进制数据，Sarvam AI应该发送JSON")

                except websockets.ConnectionClosed as e:
                    logger.bind(tag=TAG).warning(f"WebSocket连接已关闭 | Code: {e.code} | Reason: {e.reason}")
                    break
                except Exception as e:
                    logger.bind(tag=TAG).error(f"处理TTS响应时出错: {e}\n{traceback.format_exc()}")
                    break

            # Send LAST signal if session finished normally
            if session_finished:
                logger.bind(tag=TAG).debug("音频生成完成，发送LAST信号")
                self._process_before_stop_play_files()
            else:
                logger.bind(tag=TAG).warning("TTS会话结束但未收到completion信号")

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
        # Not implemented for streaming provider
        logger.bind(tag=TAG).warning("to_tts() 不支持流式TTS提供商")
        return []
