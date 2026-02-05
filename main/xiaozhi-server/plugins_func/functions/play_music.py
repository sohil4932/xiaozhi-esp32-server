import os
import re
import time
import json
import random
import difflib
import traceback
import asyncio
from pathlib import Path
from core.handle.sendAudioHandle import send_stt_message
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from core.utils.dialogue import Message
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.utils.util import audio_to_data

TAG = __name__

MUSIC_CACHE = {}

play_music_function_desc = {
    "type": "function",
    "function": {
        "name": "play_music",
        "description": "Play music, sing songs. Use when user wants to listen to music.",
        "parameters": {
            "type": "object",
            "properties": {
                "song_name": {
                    "type": "string",
                    "description": "Song name. Use 'random' if user doesn't specify. Example: 'ABCD Song' or 'random'"
                }
            },
            "required": ["song_name"],
        },
    },
}


@register_function("play_music", play_music_function_desc, ToolType.SYSTEM_CTL)
def play_music(conn: "ConnectionHandler", song_name: str):
    try:
        music_intent = f"Play music {song_name}" if song_name != "random" else "Play random music"

        if not conn.loop.is_running():
            conn.logger.bind(tag=TAG).error("Event loop not running")
            return ActionResponse(
                action=Action.RESPONSE, result="System busy", response="Please try again later"
            )

        task = conn.loop.create_task(handle_music_command(conn, music_intent))

        def handle_done(f):
            try:
                f.result()
                conn.logger.bind(tag=TAG).info("Playback complete")
            except Exception as e:
                conn.logger.bind(tag=TAG).error(f"Playback failed: {e}")

        task.add_done_callback(handle_done)

        return ActionResponse(
            action=Action.NONE, result="Command received", response="Playing music for you"
        )
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"Music error: {e}")
        return ActionResponse(
            action=Action.RESPONSE, result=str(e), response="Error playing music"
        )


def _extract_song_name(text):
    """Extract song name from user input"""
    keywords = ["play music", "play", "sing"]
    for keyword in keywords:
        if keyword in text.lower():
            parts = text.lower().split(keyword)
            if len(parts) > 1:
                return parts[1].strip()
    return None


def _find_best_match(potential_song, music_files):
    """Find best matching song"""
    best_match = None
    highest_ratio = 0

    for music_file in music_files:
        song_name = os.path.splitext(music_file)[0]
        ratio = difflib.SequenceMatcher(None, potential_song.lower(), song_name.lower()).ratio()
        if ratio > highest_ratio and ratio > 0.4:
            highest_ratio = ratio
            best_match = music_file
    return best_match


def get_music_files(music_dir, music_ext):
    music_dir = Path(music_dir)
    music_files = []
    music_file_names = []
    for file in music_dir.rglob("*"):
        if file.is_file():
            ext = file.suffix.lower()
            if ext in music_ext:
                music_files.append(str(file.relative_to(music_dir)))
                music_file_names.append(
                    os.path.splitext(str(file.relative_to(music_dir)))[0]
                )
    return music_files, music_file_names


def initialize_music_handler(conn: "ConnectionHandler"):
    global MUSIC_CACHE
    if MUSIC_CACHE == {}:
        plugins_config = conn.config.get("plugins", {})
        if "play_music" in plugins_config:
            MUSIC_CACHE["music_config"] = plugins_config["play_music"]
            MUSIC_CACHE["music_dir"] = os.path.abspath(
                MUSIC_CACHE["music_config"].get("music_dir", "./music")
            )
            MUSIC_CACHE["music_ext"] = MUSIC_CACHE["music_config"].get(
                "music_ext", (".mp3", ".wav", ".p3")
            )
            MUSIC_CACHE["refresh_time"] = MUSIC_CACHE["music_config"].get(
                "refresh_time", 60
            )
        else:
            MUSIC_CACHE["music_dir"] = os.path.abspath("./music")
            MUSIC_CACHE["music_ext"] = (".mp3", ".wav", ".p3")
            MUSIC_CACHE["refresh_time"] = 60
        MUSIC_CACHE["music_files"], MUSIC_CACHE["music_file_names"] = get_music_files(
            MUSIC_CACHE["music_dir"], MUSIC_CACHE["music_ext"]
        )
        MUSIC_CACHE["scan_time"] = time.time()
    return MUSIC_CACHE


async def handle_music_command(conn: "ConnectionHandler", text):
    initialize_music_handler(conn)
    global MUSIC_CACHE

    clean_text = re.sub(r"[^\w\s]", "", text).strip()
    conn.logger.bind(tag=TAG).debug(f"Checking music command: {clean_text}")

    if os.path.exists(MUSIC_CACHE["music_dir"]):
        if time.time() - MUSIC_CACHE["scan_time"] > MUSIC_CACHE["refresh_time"]:
            MUSIC_CACHE["music_files"], MUSIC_CACHE["music_file_names"] = (
                get_music_files(MUSIC_CACHE["music_dir"], MUSIC_CACHE["music_ext"])
            )
            MUSIC_CACHE["scan_time"] = time.time()

        potential_song = _extract_song_name(clean_text)
        if potential_song:
            best_match = _find_best_match(potential_song, MUSIC_CACHE["music_files"])
            if best_match:
                conn.logger.bind(tag=TAG).info(f"Best match found: {best_match}")
                await play_local_music(conn, specific_file=best_match)
                return True
    await play_local_music(conn)
    return True


def _get_random_play_prompt(song_name):
    """Generate random play introduction"""
    clean_name = os.path.splitext(song_name)[0]
    prompts = [
        f"Now playing {clean_name}",
        f"Playing {clean_name} for you",
        f"Here comes {clean_name}",
        f"Enjoy {clean_name}",
        f"Let's listen to {clean_name}",
        f"Now playing song {clean_name}",
        f"Playing {clean_name}",
    ]
    return random.choice(prompts)


async def play_local_music(conn: "ConnectionHandler", specific_file=None):
async def _stream_music_file_realtime(conn, music_path):
    """Stream audio file directly to ESP32 in Realtime mode

    This bypasses the TTS queue and OpenAI TTS, streaming the audio file
    directly to the client using the same low-level packet sending method
    used by the Realtime provider.

    Args:
        conn: Connection object
        music_path: Path to the music file
    """
    try:
        conn.logger.bind(tag=TAG).info(f"Converting music file to Opus format: {music_path}")

        # Convert audio file to Opus packets (16kHz, 60ms frames)
        # This uses the same conversion logic as TTS providers
        audio_packets = await audio_to_data(music_path, is_opus=True)

        if not audio_packets:
            conn.logger.bind(tag=TAG).error("Failed to convert music file to audio packets")
            return

        conn.logger.bind(tag=TAG).info(f"Streaming {len(audio_packets)} audio packets to client")

        # Pause OpenAI Realtime audio processing during music playback
        # This prevents the session from processing incoming audio and interrupting music
        if hasattr(conn, 'realtime_provider') and conn.realtime_provider:
            # Cancel any active response
            if conn.realtime_provider.response_in_progress:
                await conn.realtime_provider._cancel_response()
                conn.logger.bind(tag=TAG).info("Cancelled active OpenAI response for music playback")

            # Set flag to pause audio processing during music
            conn.realtime_provider.is_music_playing = True
            conn.logger.bind(tag=TAG).info("Paused OpenAI Realtime audio processing for music")

        # Send TTS start signal to prepare client for audio
        if conn.websocket:
            await conn.websocket.send(
                json.dumps({
                    "type": "tts",
                    "state": "start",
                    "session_id": conn.session_id
                })
            )

        # Stream each Opus packet directly to the client
        # Using the same low-level method as OpenAI Realtime provider
        for i, opus_packet in enumerate(audio_packets):
            if hasattr(conn, 'client_abort') and conn.client_abort:
                conn.logger.bind(tag=TAG).info("Music playback aborted by client")
                break

            # Send opus packet directly to client
            await conn.send_audio_to_client(opus_packet)

            # Pace sending: 50ms delay to match 60ms playback rate
            # This prevents network buffer overflow while maintaining smooth playback
            await asyncio.sleep(0.050)

            # Log progress every 100 packets
            if (i + 1) % 100 == 0:
                conn.logger.bind(tag=TAG).debug(f"Streamed {i + 1}/{len(audio_packets)} packets")

        # Send TTS stop signal when complete
        if conn.websocket:
            await conn.websocket.send(
                json.dumps({
                    "type": "tts",
                    "state": "stop",
                    "session_id": conn.session_id
                })
            )

        conn.logger.bind(tag=TAG).info("Music streaming completed")

        # Resume OpenAI Realtime audio processing after music
        if hasattr(conn, 'realtime_provider') and conn.realtime_provider:
            conn.realtime_provider.is_music_playing = False
            # Clear the input audio buffer to start fresh
            if conn.realtime_provider.ws:
                await conn.realtime_provider.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            conn.logger.bind(tag=TAG).info("Resumed OpenAI Realtime audio processing")

    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"Failed to stream music file: {str(e)}")
        conn.logger.bind(tag=TAG).error(f"Error details: {traceback.format_exc()}")

        # Make sure to resume processing even if music fails
        if hasattr(conn, 'realtime_provider') and conn.realtime_provider:
            conn.realtime_provider.is_music_playing = False
            conn.logger.bind(tag=TAG).info("Resumed OpenAI Realtime processing after error")


async def play_local_music(conn, specific_file=None):
    global MUSIC_CACHE
    try:
        if not os.path.exists(MUSIC_CACHE["music_dir"]):
            conn.logger.bind(tag=TAG).error(
                f"Music directory not found: " + MUSIC_CACHE["music_dir"]
            )
            return

        if specific_file:
            selected_music = specific_file
            music_path = os.path.join(MUSIC_CACHE["music_dir"], specific_file)
        else:
            if not MUSIC_CACHE["music_files"]:
                conn.logger.bind(tag=TAG).error("No music files found")
                return
            selected_music = random.choice(MUSIC_CACHE["music_files"])
            music_path = os.path.join(MUSIC_CACHE["music_dir"], selected_music)

        if not os.path.exists(music_path):
            conn.logger.bind(tag=TAG).error(f"Music file not found: {music_path}")
            return

        text = _get_random_play_prompt(selected_music)
        await send_stt_message(conn, text)
        conn.dialogue.put(Message(role="assistant", content=text))

        # Check if we're in Realtime mode
        if hasattr(conn, 'use_realtime') and conn.use_realtime:
            conn.logger.bind(tag=TAG).info(f"Realtime mode: Streaming music file directly - {music_path}")
            await _stream_music_file_realtime(conn, music_path)
            return

        # Standard TTS queue mode (ElevenLabs, etc.)
        if conn.intent_type == "intent_llm":
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
        conn.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=conn.sentence_id,
                sentence_type=SentenceType.MIDDLE,
                content_type=ContentType.TEXT,
                content_detail=text,
            )
        )
        conn.tts.tts_text_queue.put(
            TTSMessageDTO(
                sentence_id=conn.sentence_id,
                sentence_type=SentenceType.MIDDLE,
                content_type=ContentType.FILE,
                content_file=music_path,
            )
        )
        if conn.intent_type == "intent_llm":
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )

    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"Music playback failed: {str(e)}")
        conn.logger.bind(tag=TAG).error(f"Error details: {traceback.format_exc()}")