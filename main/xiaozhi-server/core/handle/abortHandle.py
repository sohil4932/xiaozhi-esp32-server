import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
TAG = __name__


async def handleAbortMessage(conn: "ConnectionHandler"):
    conn.logger.bind(tag=TAG).info("Abort message received")

    # For realtime mode - disconnect the realtime provider completely
    if hasattr(conn, 'use_realtime') and conn.use_realtime and hasattr(conn, 'realtime_provider') and conn.realtime_provider:
        conn.logger.bind(tag=TAG).info("Realtime mode detected - disconnecting realtime provider")
        try:
            await conn.realtime_provider.cleanup()
            conn.logger.bind(tag=TAG).info("Realtime provider disconnected on abort")
        except Exception as e:
            conn.logger.bind(tag=TAG).error(f"Error disconnecting realtime provider: {e}")
        return

    # For regular mode (ASR/TTS pipeline)
    # 设置成打断状态，会自动打断llm、tts任务
    conn.client_abort = True
    conn.clear_queues()
    # 打断客户端说话状态
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    conn.logger.bind(tag=TAG).info("Abort message received-end")
