# Memory Provider Audio Integration

## Overview

This document describes how to integrate audio data with memory providers (PowerMem, Mem0, etc.) to enable audio-aware memory systems that can store and retrieve conversation audio alongside text.

## Current Architecture

### How Memory Saves Data (Text Only)

```python
# connection.py: memory_save_thread
async def memory_save_thread(self):
    while not self.stop_event.is_set():
        # Wait for dialogue to be ready
        await asyncio.sleep(5)

        # Get snapshot of dialogue history
        dialogue_snapshot = list(self.dialogue_history)

        # Save to memory provider (text only)
        await self.memory.save_memory(dialogue_snapshot, self.session_id)
```

### Current Message Format

```python
# dialogue.py: Message class
class Message:
    role: str       # "user" or "assistant"
    content: str    # Text content (or JSON with metadata)
```

## Problem: Audio Not Available to Memory Providers

Currently, when memory providers receive messages:
- ✅ They get text content
- ❌ They don't get audio data
- ❌ Can't store audio URLs/references
- ❌ Can't implement audio-based retrieval

**Why?** Audio is reported separately to chat history, not passed to memory providers.

## Solution: Extend Message Class with Audio References

### Step 1: Enhance Message Class

```python
# core/utils/dialogue.py

class Message:
    """Message with optional audio reference"""

    def __init__(
        self,
        role: str,
        content: str,
        audio_id: str = None,      # NEW: Reference to audio in database
        audio_url: str = None,      # NEW: URL to access audio
        timestamp: int = None       # NEW: Message timestamp
    ):
        self.role = role
        self.content = content
        self.audio_id = audio_id
        self.audio_url = audio_url
        self.timestamp = timestamp or int(time.time() * 1000)

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "role": self.role,
            "content": self.content,
            "audio_id": self.audio_id,
            "audio_url": self.audio_url,
            "timestamp": self.timestamp
        }

    def get_text_only(self):
        """Get text content without audio metadata (for LLM)"""
        # Extract text from JSON format if present
        if self.content and self.content.strip().startswith("{"):
            try:
                data = json.loads(self.content)
                if "content" in data:
                    return data["content"]
            except:
                pass
        return self.content
```

### Step 2: Store Audio ID When Reporting

```python
# core/handle/reportHandle.py

def enqueue_asr_report(conn: "ConnectionHandler", text, opus_data):
    """Queue ASR audio for reporting and store audio_id"""
    if not conn.read_config_from_api or conn.chat_history_conf == 0:
        return

    # Queue audio for upload
    if conn.chat_history_conf == 2:
        # Store audio_id placeholder that will be filled after upload
        audio_id_future = asyncio.Future()
        conn.report_queue.put((1, text, opus_data, int(time.time()), audio_id_future))

        # Attach future to last message so we can update it later
        if conn.dialogue_history:
            last_msg = conn.dialogue_history[-1]
            if last_msg.role == "user":
                last_msg._audio_id_future = audio_id_future
    else:
        conn.report_queue.put((1, text, None, int(time.time()), None))


async def report(conn, type, text, opus_data, report_time, audio_id_future=None):
    """Execute chat history report with audio"""
    try:
        if opus_data:
            audio_data = opus_to_wav(conn, opus_data)
        else:
            audio_data = None

        # Upload to Management API
        response = await manage_report(
            mac_address=conn.device_id,
            session_id=conn.session_id,
            chat_type=type,
            content=text,
            audio=audio_data,
            report_time=report_time,
        )

        # Extract audio_id from response
        if response and "audioId" in response and audio_id_future:
            audio_id = response["audioId"]
            audio_id_future.set_result(audio_id)
            conn.logger.bind(tag=TAG).debug(f"Audio uploaded, audioId: {audio_id}")

    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"Chat history report failed: {e}")
        if audio_id_future:
            audio_id_future.set_exception(e)
```

### Step 3: Populate Audio IDs Before Memory Save

```python
# connection.py: memory_save_thread

async def memory_save_thread(self):
    while not self.stop_event.is_set():
        await asyncio.sleep(5)

        # Get snapshot
        dialogue_snapshot = list(self.dialogue_history)

        # Wait for audio IDs to be populated (if audio reporting is enabled)
        if self.read_config_from_api and self.chat_history_conf == 2:
            await self._populate_audio_ids(dialogue_snapshot)

        # Save to memory (now with audio references)
        await self.memory.save_memory(dialogue_snapshot, self.session_id)


async def _populate_audio_ids(self, messages):
    """Wait for audio upload to complete and populate audio_id fields"""
    for msg in messages:
        if hasattr(msg, '_audio_id_future') and msg._audio_id_future:
            try:
                # Wait for audio upload (with timeout)
                audio_id = await asyncio.wait_for(msg._audio_id_future, timeout=10.0)
                msg.audio_id = audio_id

                # Generate audio URL
                if self.read_config_from_api:
                    base_url = self.config.get("management_api_url", "https://core.nokotoys.com/xiaozhi")
                    msg.audio_url = f"{base_url}/agent/chat-audio/{audio_id}"

                self.logger.bind(tag=TAG).debug(f"Populated audio_id: {audio_id} for message")
            except asyncio.TimeoutError:
                self.logger.bind(tag=TAG).warning("Timeout waiting for audio upload")
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"Error populating audio_id: {e}")
            finally:
                del msg._audio_id_future  # Clean up
```

### Step 4: Update Memory Base Class

```python
# core/providers/memory/base.py

class MemoryProviderBase(ABC):
    @abstractmethod
    async def save_memory(self, msgs, session_id=None):
        """Save messages with optional audio references

        Args:
            msgs: List of Message objects with:
                  - role: str
                  - content: str
                  - audio_id: Optional[str]
                  - audio_url: Optional[str]
                  - timestamp: Optional[int]
            session_id: Session identifier

        Returns:
            Result from memory provider
        """
        pass
```

### Step 5: Update PowerMem Implementation

```python
# core/providers/memory/powermem/powermem.py

async def save_memory(self, msgs, session_id=None):
    """Save conversation with audio metadata to PowerMem"""
    if not self.use_powermem or self.memory_client is None:
        return None

    if len(msgs) < 2:
        return None

    try:
        messages = []
        for message in msgs:
            if message.role == "system":
                continue

            # Extract text content
            content = self._extract_text_content(message.content)

            # Build message with metadata
            msg_dict = {
                "role": message.role,
                "content": content
            }

            # Add audio metadata if available
            if hasattr(message, 'audio_id') and message.audio_id:
                msg_dict["metadata"] = {
                    "audio_id": message.audio_id,
                    "audio_url": getattr(message, 'audio_url', None),
                    "timestamp": getattr(message, 'timestamp', None)
                }

            messages.append(msg_dict)

        # Save to PowerMem with audio metadata
        result = self.memory_client.add(
            messages=messages,
            user_id=self.role_id,
            metadata={
                "session_id": session_id,
                "has_audio": any(hasattr(m, 'audio_id') and m.audio_id for m in msgs)
            }
        )

        if asyncio.iscoroutine(result):
            result = await result

        logger.bind(tag=TAG).debug(f"Saved memory with audio metadata: {result}")
        return result

    except Exception as e:
        logger.bind(tag=TAG).error(f"Error saving memory with audio: {e}")
        return None


def _extract_text_content(self, content):
    """Extract text from JSON format if present"""
    if content and content.strip().startswith("{"):
        try:
            data = json.loads(content)
            if "content" in data:
                return data["content"]
        except:
            pass
    return content
```

### Step 6: Query with Audio Support

```python
# core/providers/memory/powermem/powermem.py

async def query_memory(self, query: str) -> str:
    """Query memories and include audio URLs if available"""
    if not self.use_powermem or self.memory_client is None:
        return ""

    try:
        search_query = self._extract_text_content(query)

        # Search memories
        results = await self.memory_client.search(
            query=search_query,
            user_id=self.role_id,
            limit=30
        )

        if results and "results" in results:
            memories = []
            for entry in results.get("results", []):
                memory = entry.get("memory", "") or entry.get("content", "")
                metadata = entry.get("metadata", {})

                # Format with audio link if available
                if metadata.get("audio_url"):
                    memory_str = f"{memory} [🔊 Audio]({metadata['audio_url']})"
                else:
                    memory_str = memory

                timestamp = metadata.get("timestamp", "")
                if timestamp:
                    memories.append((timestamp, f"[{timestamp}] {memory_str}"))
                else:
                    memories.append(("", memory_str))

            # Sort and format
            memories.sort(key=lambda x: x[0], reverse=True)
            if memories:
                return "\n".join(f"- {m[1]}" for m in memories)

        return ""

    except Exception as e:
        logger.bind(tag=TAG).error(f"Error querying memory: {e}")
        return ""
```

## Complete Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│  1. User Speaks                                                         │
│     ESP32 → Opus audio → Xiaozhi Server                                │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  2. ASR Processing                                                      │
│     - Decode audio                                                      │
│     - Speech to text: "你好，我是爸爸"                                  │
│     - Add to dialogue: Message(role="user", content="你好...")          │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ├──────────────────────────────────────┐
                         │                                      │
                         ▼                                      ▼
┌────────────────────────────────────────┐  ┌──────────────────────────┐
│  3a. Report to Chat History            │  │  3b. LLM Processing      │
│      (Background Thread)                │  │      (Main Thread)       │
│                                         │  │                          │
│  - Queue: (text, opus_data, future)   │  │  - Generate response     │
│  - Upload audio → Management API       │  │  - Add to dialogue       │
│  - Get audioId from response           │  │  - Stream TTS            │
│  - Set future.result(audioId)          │  └──────────────────────────┘
└────────────────────────┬───────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  4. Memory Save (Every 5 seconds)                                      │
│                                                                         │
│  - Get dialogue snapshot                                               │
│  - Wait for audio_id futures (timeout 10s)                            │
│  - Populate Message.audio_id, Message.audio_url                       │
│  - Call memory.save_memory(messages, session_id)                      │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  5. PowerMem Stores Memory + Audio                                     │
│                                                                         │
│  PowerMem.add({                                                        │
│    messages: [                                                          │
│      {                                                                  │
│        role: "user",                                                   │
│        content: "你好，我是爸爸",                                      │
│        metadata: {                                                     │
│          audio_id: "audio_123",                                        │
│          audio_url: "https://core.nokotoys.com/.../audio_123",        │
│          timestamp: 1709452800000                                      │
│        }                                                                │
│      }                                                                  │
│    ],                                                                   │
│    user_id: "ESP32_ABCD1234"                                           │
│  })                                                                     │
└────────────────────────────────────────────────────────────────────────┘
```

## Benefits

### 1. Audio-Aware Memory Retrieval

PowerMem can now:
- ✅ Store audio URLs with memories
- ✅ Retrieve relevant memories with audio
- ✅ Enable playback of historical conversations
- ✅ Support multimodal memory queries (future)

### 2. Backwards Compatible

- ✅ Works with existing memory providers (audio fields optional)
- ✅ If `audio_id` is None, behavior is unchanged
- ✅ Only activates when `chat_history_conf == 2`

### 3. Future Extensions

Once audio is in memory:
- 🔮 Audio-based similarity search
- 🔮 Emotion analysis from audio
- 🔮 Voice cloning for personalized TTS
- 🔮 Audio summarization

## Configuration

### Enable Audio in Memory

```yaml
# config.yaml

# Enable chat history with audio
read_config_from_api: true

# Management API will set:
# chat_history_conf: 2  (text + audio)

# Memory provider with audio support
memory:
  type: "powermem"
  enable_user_profile: true
  database_provider: "sqlite"
  llm_provider: "qwen"
  llm_api_key: "YOUR_KEY"
  embedding_provider: "qwen"
  embedding_api_key: "YOUR_KEY"
```

## Example: Querying Memory with Audio

### User Query
```
User: "What did I say about my father?"
```

### PowerMem Response
```
【相关记忆】
- [2026-03-03 10:30:00] "你好，我是爸爸" [🔊 Audio](https://core.nokotoys.com/.../audio_123)
- [2026-03-03 10:31:15] "我爸爸喜欢喝茶" [🔊 Audio](https://core.nokotoys.com/.../audio_456)
```

### Agent Response
```
Agent: "You mentioned that you are the father at 10:30, and later said
       your father likes drinking tea. Would you like me to play back
       those conversations?"
```

## Implementation Checklist

- [ ] Update `Message` class with audio fields
- [ ] Modify `enqueue_asr_report` to return audio_id future
- [ ] Update `report()` to populate audio_id future
- [ ] Add `_populate_audio_ids()` method to connection
- [ ] Update `memory_save_thread` to wait for audio IDs
- [ ] Update `MemoryProviderBase` documentation
- [ ] Update `PowerMemProvider.save_memory()` with audio support
- [ ] Update `PowerMemProvider.query_memory()` to show audio URLs
- [ ] Add Management API endpoint for audio playback
- [ ] Test with PowerMem
- [ ] Test with Mem0
- [ ] Document breaking changes (if any)

## Testing

### Test Audio Storage
```python
# After user speaks
assert message.audio_id is not None
assert message.audio_url.startswith("https://")
```

### Test Memory Retrieval
```python
result = await memory.query_memory("what did I say")
assert "[🔊 Audio]" in result
```

### Test Backwards Compatibility
```python
# With audio disabled (chat_history_conf != 2)
assert message.audio_id is None
# Memory still works without audio
```

## Migration Path

### Phase 1: Audio Metadata (This Design)
- Store audio URLs in metadata
- No breaking changes

### Phase 2: Audio Embeddings (Future)
- Extract audio features
- Store in vector database
- Enable audio similarity search

### Phase 3: Multimodal Memory (Future)
- Audio + Text + Image
- Cross-modal retrieval
- Unified representation

## Summary

This design enables memory providers to access audio data while maintaining backwards compatibility:

✅ **Zero Breaking Changes**
- Existing code continues to work
- Audio fields are optional
- Only activates when configured

✅ **Universal Support**
- Works with any memory provider
- PowerMem, Mem0, custom providers
- Just check for `audio_id` field

✅ **Future-Proof**
- Foundation for audio embeddings
- Enables multimodal memory
- Extensible metadata structure

**Answer to your question:** Yes! Any memory provider can access audio by checking `message.audio_id` and `message.audio_url` fields. If they're None, it works like before. If they're populated, the provider can store/use them however it wants.
