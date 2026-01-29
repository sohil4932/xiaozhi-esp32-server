# Hume.ai EVI (Empathic Voice Interface) Provider

This document provides information about using Hume.ai's EVI as a realtime provider in xiaozhi-esp32-server.

## Overview

Hume.ai EVI is an emotionally intelligent voice AI that combines ASR (Automatic Speech Recognition), LLM (Language Model), and TTS (Text-to-Speech) with emotional intelligence and prosody analysis in a single low-latency WebSocket connection.

### Key Features

- **Emotional Intelligence**: Detects and responds to emotional cues in user's voice
- **Vocal Expression Measurement**: Real-time analysis of user emotions during conversation
- **Always Interruptible**: Stops immediately when user speaks
- **Multi-language Support**: 11 languages supported (en, zh, es, fr, de, it, pt, ja, ko, hi, ar)
- **Prosody-Aware Speech**: Natural, emotionally expressive voice synthesis
- **Configurable LLM Backend**: Choose from Anthropic, OpenAI, Google, etc.
- **Custom Voice Design**: Create unique voice personalities via Hume console

### Advantages Over Other Providers

| Feature | Hume EVI | OpenAI Realtime | Gemini Live |
|---------|----------|-----------------|-------------|
| Emotion Detection | ✅ Yes | ❌ No | ❌ No |
| Prosody Analysis | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| Voice Customization | ✅ Web Console | ⚠️ Limited | ⚠️ Limited |
| Interruption Handling | ✅ Excellent | ✅ Good | ✅ Good |
| Multi-language | ✅ 11 languages | ✅ Many | ✅ Many |
| Expected Latency | 300-500ms | 300-500ms | 300-500ms |

## Configuration

### 1. Get API Key

1. Sign up at [Hume.ai Platform](https://platform.hume.ai/)
2. Navigate to [API Keys](https://platform.hume.ai/settings/keys)
3. Create a new API key

### 2. Configure xiaozhi-server

Edit `config.yaml`:

```yaml
# In selected_module section:
selected_module:
  Realtime: HumeEVI  # Select Hume EVI provider

# In Realtime section:
Realtime:
  HumeEVI:
    type: hume_realtime
    api_key: YOUR-HUME-API-KEY-HERE  # Required
    config_id: ""  # Optional: EVI configuration ID
    language: en  # Language code
    temperature: 0.8  # Model temperature (0.0-1.0)
```

### 3. Optional: Create Custom EVI Configuration

For advanced customization (voice, personality, LLM backend):

1. Go to [Hume Platform Configs](https://platform.hume.ai/settings/configs)
2. Create a new EVI configuration
3. Customize:
   - **Voice & Personality**: Choose voice characteristics, speaking style
   - **System Prompt**: Define AI personality and behavior
   - **LLM Backend**: Select Anthropic Claude, OpenAI GPT, Google Gemini, etc.
   - **Emotional Style**: Configure emotional expression intensity
4. Copy the `config_id`
5. Add to `config.yaml`:
   ```yaml
   config_id: "your-config-id-here"
   ```

### 4. Configure Tools (For Function Calling)

**✨ AUTOMATIC (Recommended - Zero Config!)**

Just leave `config_id` empty:
```yaml
HumeEVI:
  api_key: YOUR-HUME-API-KEY
  config_id: ""  # Empty = auto-config!
```

xiaozhi-server automatically:
- ✅ Creates Hume config with ALL your tools
- ✅ Uses Claude 3.5 Sonnet
- ✅ Includes system prompt
- ✅ Logs config_id for reuse

**🛠️ MANUAL (Advanced Control)**

1. Go to [Hume Platform](https://platform.hume.ai/settings/configs)
2. Create EVI config
3. Add tools with JSON schema
4. Choose Claude/GPT/Gemini/Moonshot
5. Copy `config_id` → `config.yaml`

**👨‍💻 PROGRAMMATIC API**

```python
from core.providers.realtime.hume_config_manager import HumeConfigManager

manager = HumeConfigManager("YOUR_API_KEY")
config_id = await manager.create_config_with_tools(
    name="My Config",
    tools=my_tools
)
```

## Audio Format

Hume EVI uses the following audio format (matching ESP32):

- **Encoding**: Linear PCM16 (16-bit signed integers)
- **Sample Rate**: 16kHz
- **Channels**: 1 (Mono)
- **Frame Size**: 960 samples (60ms)
- **Transmission**: Base64-encoded over WebSocket

## WebSocket Protocol

### Connection

```
wss://api.hume.ai/v0/assistant/chat?api_key=YOUR_API_KEY&config_id=YOUR_CONFIG_ID
```

### Messages

**Sending Audio (Client → Hume)**
```json
{
  "type": "audio_input",
  "data": "<base64-encoded PCM16>"
}
```

**Receiving Audio (Hume → Client)**
```json
{
  "type": "audio_output",
  "data": "<base64-encoded WAV file>"
}
```

**User Transcript**
```json
{
  "type": "user_message",
  "message": {
    "content": "What is my favorite food?"
  }
}
```

**Assistant Response**
```json
{
  "type": "assistant_message",
  "message": {
    "content": "Your favorite food is pizza!"
  }
}
```

**User Interruption**
```json
{
  "type": "user_interruption"
}
```

**Assistant End**
```json
{
  "type": "assistant_end"
}
```

**Tool Call (Hume → Server)**
```json
{
  "type": "tool_call",
  "tool_type": "function",
  "tool_call_id": "call_xyz123",
  "name": "get_weather",
  "parameters": "{\"location\":\"New York\"}"
}
```

**Tool Response (Server → Hume)**
```json
{
  "type": "tool_response",
  "tool_call_id": "call_xyz123",
  "content": "72°F, sunny"
}
```

## Integration with Existing Features

### ✅ Supported Features

- **Memory System**: Short-term and long-term memory integration ✅
- **Function Calling**: Full native support for MCP tools, plugins, device control ✅
- **Device MCP**: Control device settings (volume, brightness, etc.) ✅
- **Music Playback**: Play music function integration ✅
- **Chat History**: Automatic logging to management API ✅
- **Multi-session**: Proper session isolation ✅
- **Interruption Handling**: User can interrupt at any time ✅
- **Exit Intent**: Proper connection closing on goodbye ✅

### ⚠️ Important Configuration Notes

- **Tool Pre-configuration Required**: Tools must be configured in your Hume EVI config (via web console or API) **before** starting the session. Unlike OpenAI/Gemini where tools are sent dynamically, Hume requires tools to be part of your EVI configuration.
- **Supported LLMs for Tools**: Tool use only works with Claude, GPT, Gemini, or Moonshot AI as the supplemental LLM in your EVI config
- **Voice & Personality**: Best configured via Hume web console for optimal emotional intelligence

## Troubleshooting

### Issue: "Cannot connect to Hume EVI"

**Solution**: Check:
1. API key is valid
2. Network connectivity
3. API quota not exceeded

### Issue: "No audio output"

**Solution**: Check:
1. Hume console shows active session
2. Audio format settings are correct
3. Client is sending audio properly

### Issue: "Functions not working"

**Solution**:
1. Ensure functions are defined in system prompt instructions
2. Configure EVI personality to use function calling via Hume console
3. Test with simple commands first

### Issue: "Voice sounds robotic"

**Solution**:
1. Create custom EVI config in Hume console
2. Adjust emotional expressiveness settings
3. Try different voice configurations

## Pricing

Contact Hume.ai for pricing information. Pricing typically includes:
- Per-minute conversation charges
- API call limits based on tier
- Enterprise plans available

## Comparison with Other Providers

### When to Use Hume EVI

✅ **Use Hume when you need:**
- Emotional intelligence and empathy in responses
- Natural, emotionally expressive voice
- Real-time emotion detection from user voice
- Healthcare, therapy, education, or customer service applications
- Maximum interruption responsiveness

❌ **Don't use Hume if:**
- You need native function calling protocol (use OpenAI)
- You want all configuration in config.yaml (use OpenAI/Gemini)
- Budget is primary concern (OpenAI may be cheaper)

### When to Use OpenAI Realtime

✅ **Use OpenAI when you need:**
- Native function calling with structured outputs
- Full control via code (no web console needed)
- Wide model selection (GPT-4o variants)
- Cost-effective high-volume usage

### When to Use Gemini Live

✅ **Use Gemini when you need:**
- Best echo cancellation and VAD
- Native 16kHz support (no resampling)
- Free tier for testing
- Integration with Google Cloud ecosystem

## Resources

- [Hume.ai Platform](https://platform.hume.ai/)
- [EVI Documentation](https://dev.hume.ai/docs/speech-to-speech-evi/overview)
- [API Reference](https://dev.hume.ai/reference/speech-to-speech-evi)
- [Audio Guide](https://dev.hume.ai/docs/speech-to-speech-evi/guides/audio)
- [Community Examples](https://github.com/HumeAI/hume-api-examples)

## Support

For issues specific to:
- **Hume EVI API**: Contact Hume.ai support or check their documentation
- **xiaozhi-server integration**: Open an issue in the xiaozhi-esp32-server repository
