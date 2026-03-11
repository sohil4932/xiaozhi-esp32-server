import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
TAG = __name__


async def handleAbortMessage(conn: "ConnectionHandler"):
    conn.logger.bind(tag=TAG).info("Abort message received")

    # For realtime mode
    if hasattr(conn, 'use_realtime') and conn.use_realtime and hasattr(conn, 'realtime_provider') and conn.realtime_provider:
        provider = conn.realtime_provider

        # 1) If music is playing → stop music, stay connected
        if getattr(provider, 'is_music_playing', False):
            conn.logger.bind(tag=TAG).info("Abort: stopping music playback")
            conn.client_abort = True
            provider.is_music_playing = False
            try:
                await conn.websocket.send(
                    json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
                )
            except Exception:
                pass
            conn.logger.bind(tag=TAG).info("Music stopped - ready to listen")
            conn.client_abort = False
            return

        # 2) If already aborted (second tap while in standby) → disconnect
        if getattr(provider, 'user_aborted', False):
            conn.logger.bind(tag=TAG).info("Abort: second tap - disconnecting provider")
            try:
                await provider.cleanup()
                conn.logger.bind(tag=TAG).info("Provider disconnected")
            except Exception as e:
                conn.logger.bind(tag=TAG).error(f"Error disconnecting provider: {e}")
            return

        # 3) First tap → stop agent speech, go to standby
        conn.logger.bind(tag=TAG).info("Abort: stopping agent - going to standby")
        conn.client_abort = True
        if hasattr(provider, 'audio_output_blocked'):
            provider.audio_output_blocked = True
        if hasattr(provider, 'user_aborted'):
            provider.user_aborted = True
        if hasattr(provider, '_output_pcm_buffer'):
            provider._output_pcm_buffer = b""
        if hasattr(provider, 'audio_session_started'):
            provider.audio_session_started = False
            provider.audio_frames_sent = 0
        try:
            await conn.websocket.send(
                json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
            )
        except Exception:
            pass
        conn.logger.bind(tag=TAG).info("Agent stopped - tap again to disconnect, or speak to resume")
        return

    # For regular mode (ASR/TTS pipeline)
    conn.client_abort = True
    conn.clear_queues()
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    conn.logger.bind(tag=TAG).info("Abort message received-end")
