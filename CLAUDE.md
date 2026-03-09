# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**xiaozhi-esp32-server** is a comprehensive backend system for ESP32-based voice assistant hardware. It provides:
- Real-time voice interaction (ASR → LLM → TTS pipeline)
- Multi-user management with voice print recognition
- Plugin-based tool system (IoT, MCP, custom functions)
- Web-based management console
- Support for 30+ AI service providers

This is a multi-component distributed system with Python (core AI engine), Java Spring Boot (management API), and Vue.js (web/mobile frontends).

## Development Commands

### Python Server (xiaozhi-server)

```bash
# Navigate to server directory
cd main/xiaozhi-server

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py

# Run performance tests
python performance_tester.py

# Test specific modules (ASR/LLM/TTS)
# First configure API keys in data/.config.yaml
# Then run performance tester and select module
```

### Java Backend (manager-api)

```bash
cd main/manager-api

# Build with Maven
mvn clean install

# Run Spring Boot application
mvn spring-boot:run
```

### Web Frontend (manager-web)

```bash
cd main/manager-web

# Install dependencies
npm install

# Run development server
npm run serve

# Build for production
npm run build
```

### Mobile Frontend (manager-mobile)

```bash
cd main/manager-mobile

# Install dependencies
pnpm install

# Run development
pnpm run dev
```

### Docker Deployment

```bash
# Single server mode (minimal)
docker compose up -d

# Full stack mode (all modules)
cd main/xiaozhi-server
docker compose -f docker-compose_all.yml up -d

# View logs
docker logs -f xiaozhi-esp32-server
```

## Architecture Overview

### System Communication Flow

```
ESP32 Device (WebSocket)
    ↓
xiaozhi-server (Python)
    ├── VAD (Voice Activity Detection) → Silero
    ├── ASR (Speech Recognition) → FunASR/Xunfei/Cloud APIs
    ├── LLM (Language Model) → OpenAI/Qwen/GLM/Custom
    ├── Memory → Local/Mem0ai/PowerMem
    ├── Intent Recognition → Function Call/LLM-based
    ├── Tool Execution → Plugins/IoT/MCP
    └── TTS (Text-to-Speech) → EdgeTTS/Volcano/Cloud APIs
    ↓
Audio Response → ESP32 Device
```

### Multi-Module Architecture

**xiaozhi-server** (Python - Core AI Engine)
- Location: `main/xiaozhi-server/`
- Entry: `app.py`
- Handles: WebSocket server, audio processing, AI pipeline
- Config: `config.yaml` (template), `data/.config.yaml` (active)
- Key directories:
  - `core/`: Core processing logic
    - `handle/`: Message handlers (audio, text, intent)
    - `providers/`: AI service providers (ASR, LLM, TTS, Memory, etc.)
    - `api/`: HTTP endpoints (OTA, vision)
  - `plugins_func/`: Extensible function plugins
  - `config/`: Configuration and logging
  - `models/`: Local AI models (FunASR, Silero VAD)

**manager-api** (Java Spring Boot - Management Backend)
- Location: `main/manager-api/`
- Purpose: User/device management, configuration storage
- Database: MySQL (stores users, devices, AI configs)
- Provides REST APIs for manager-web/mobile

**manager-web** (Vue.js - Web Console)
- Location: `main/manager-web/`
- Purpose: Web-based admin interface for configuration

**manager-mobile** (uni-app + Vue3 - Mobile Console)
- Location: `main/manager-mobile/`
- Purpose: Cross-platform mobile admin interface

## Configuration System

### Configuration Priority
1. `data/.config.yaml` (user overrides - highest priority)
2. `config.yaml` (default template)
3. Manager API settings (when `read_config_from_api: true`)

### Critical Configuration Sections

**Server Settings** (`server:`)
- `ip`, `port`: WebSocket server binding
- `http_port`: OTA and vision API port
- `auth.enabled`: Enable device authentication
- `mqtt_gateway`, `udp_gateway`: IoT gateways

**Module Selection** (`selected_module:`)
```yaml
VAD: SileroVAD           # Voice activity detection
ASR: FunASR              # Speech recognition (or XunfeiStreamASR)
LLM: ChatGLMLLM          # Language model (or OpenAILLM)
VLLM: ChatGLMVLLM        # Vision model
TTS: EdgeTTS             # Text-to-speech (or HuoshanDoubleStreamTTS)
Memory: mem_local_short  # Conversation memory
Intent: function_call    # Intent recognition
Realtime: OpenAIRealtime # Optional: Skip ASR+LLM+TTS pipeline
```

**Plugin Functions** (`Intent.function_call.functions:`)
```yaml
functions:
  - get_weather          # Weather queries
  - get_news_from_newsnow # News fetching
  - play_music           # Music playback
  - change_role          # Character switching
  - search_from_ragflow  # Knowledge base RAG
```

## Provider Architecture

All AI providers follow a consistent base class pattern in `core/providers/`:

### ASR Providers (`asr/`)
- Base: `base.py` - Abstract `ASRProvider` class
- Implementations: `fun_local.py`, `xunfei_stream.py`, `doubao_stream.py`, etc.
- Stream vs Batch: Stream providers process audio chunks in real-time

### LLM Providers (`llm/`)
- Base: `base.py` - Abstract `LLMProvider` class
- Implementations: `openai/openai.py`, `ollama/ollama.py`, `dify/dify.py`, etc.
- Support: Function calling, streaming responses

### TTS Providers (`tts/`)
- Base: `base.py` - Abstract `TTSProvider` class
- Implementations: `edge.py`, `huoshan_double_stream.py`, `elevenlabs_stream.py`, etc.
- Stream providers: Send audio chunks progressively

### Memory Providers (`memory/`)
- `mem_local_short`: Local short-term memory with LLM summarization
- `mem0ai`: Cloud-based memory service
- `powermem`: OceanBase-backed intelligent memory with user profiling
- `nomem`: Disable memory

### Realtime Providers (`realtime/`)
- `openai_realtime.py`: OpenAI Realtime API (ASR+LLM+TTS in one)
- `gemini_live.py`: Google Gemini Live API
- `hume_realtime.py`: Hume.ai EVI with emotion detection
- Replaces traditional pipeline with single WebSocket connection (~300-500ms latency)

## Tool/Plugin System

### Plugin Location and Structure
- Location: `plugins_func/functions/`
- Registration: `plugins_func/register.py`
- Loading: `plugins_func/loadplugins.py`

### Plugin Types

**Server Plugins** (`tools/server_plugins/`)
- Functions defined in `plugins_func/functions/`
- Examples: `get_weather.py`, `play_music.py`, `search_from_ragflow.py`
- Execute locally on server

**Device IoT Tools** (`tools/device_iot/`)
- Execute on ESP32 device (volume control, LED, etc.)
- Uses custom IoT protocol

**Device MCP Tools** (`tools/device_mcp/`)
- MCP protocol tools running on ESP32
- Accessed via device WebSocket

**Server MCP Tools** (`tools/server_mcp/`)
- MCP servers running on backend
- Configured in `mcp_server_settings.json`

**MCP Endpoint** (`tools/mcp_endpoint/`)
- External MCP server integration
- Configured via `mcp_endpoint` URL

### Adding New Plugins

1. Create `plugins_func/functions/your_plugin.py`
2. Define plugin metadata:
```python
def register():
    return {
        "name": "your_function_name",
        "description": "What it does",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

async def execute(arguments: dict, context: dict) -> dict:
    # Implementation
    return {"result": "success"}
```
3. Add to `config.yaml` under `Intent.function_call.functions`

## Message Handler System

### Handler Registry Pattern
Located in `core/handle/textHandler/`:
- `textMessageType.py`: Defines message type enum
- `textMessageHandler.py`: Abstract base handler
- `textMessageHandlerRegistry.py`: Handler registration
- Specific handlers: `helloMessageHandler.py`, `iotMessageHandler.py`, etc.

### Message Flow
1. WebSocket receives message → `core/connection.py`
2. Route to handler based on type → `textMessageProcessor.py`
3. Execute handler logic → Specific handler (e.g., `receiveAudioHandle.py`)
4. Send response back to device

## Audio Processing Pipeline

### Traditional Pipeline (ASR → LLM → TTS)
1. **VAD** detects speech activity
2. **ASR** converts audio → text
3. **Voiceprint** (optional) identifies speaker
4. **Memory** retrieves context
5. **LLM** generates response (with tool calling)
6. **Intent** recognizes and executes tools
7. **TTS** synthesizes speech
8. Audio chunks sent to device

### Realtime API Mode
When `Realtime: OpenAIRealtime` is set:
- Single WebSocket to provider
- Replaces ASR+LLM+TTS with unified API
- ~2.5 second latency reduction
- Still supports tools/plugins via function calling

## Key Files Reference

### Configuration
- `main/xiaozhi-server/config.yaml` - Default config template
- `main/xiaozhi-server/data/.config.yaml` - User overrides (gitignored)
- `main/xiaozhi-server/agent-base-prompt.txt` - System prompt template

### Core Entry Points
- `main/xiaozhi-server/app.py:46` - Main server startup
- `main/xiaozhi-server/core/websocket_server.py` - WebSocket server
- `main/xiaozhi-server/core/http_server.py` - HTTP/OTA server
- `main/xiaozhi-server/core/connection.py` - Client connection handler

### Provider Factories
- `main/xiaozhi-server/config/settings.py` - Provider initialization
- `main/xiaozhi-server/core/providers/*/base.py` - Base classes

### Tool System
- `main/xiaozhi-server/core/providers/tools/unified_tool_manager.py` - Unified tool orchestration
- `main/xiaozhi-server/core/providers/tools/unified_tool_handler.py` - Tool execution coordination

## Testing

### Audio Interaction Test
```bash
# Use Chrome browser to open:
main/xiaozhi-server/test/test_page.html
```

### Performance Testing
```bash
cd main/xiaozhi-server
python performance_tester.py
# Tests ASR, LLM, VLLM, TTS response times
# Only tests providers with configured API keys
```

### Vision Model Test
```bash
# Configure VLLM provider first
# Upload image via test interface
# Tests multimodal capabilities
```

## Important Development Notes

### When Modifying Providers
1. All providers must inherit from respective base class
2. Implement all abstract methods
3. Handle errors gracefully (return error dict, don't raise)
4. Support both sync and async patterns where needed
5. Add configuration template to `config.yaml`

### When Adding Features
1. Check if plugin system is appropriate first
2. Avoid modifying core pipeline unless necessary
3. Use dependency injection via `config` dict
4. Follow existing logging patterns with `logger.bind(tag=TAG)`

### When Debugging
1. Check logs in `tmp/server.log`
2. Set `log_level: DEBUG` in config
3. Use WebSocket test page for direct testing
4. Performance tester for provider-specific issues

### Configuration Changes
- Never commit `data/.config.yaml` (contains API keys)
- Always update `config.yaml` template for new features
- Document new config options in deployment docs

## Common Patterns

### Provider Initialization
```python
from core.providers.asr.base import ASRProvider

class MyASR(ASRProvider):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key")
        # Initialize provider
```

### Async Tool Execution
```python
async def execute(arguments: dict, context: dict) -> dict:
    try:
        # Access config via context
        config = context.get("config", {})
        # Perform operation
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### Streaming Audio
```python
async def synthesize_streaming(self, text: str):
    async for chunk in self.generate_audio(text):
        yield chunk  # Yield audio chunks as they're generated
```

## Deployment Modes

### Minimal Mode (Single Container)
- Only `xiaozhi-server` runs
- Config stored in files (no database)
- For low-resource environments (2-core, 2GB RAM)

### Full Mode (Multi-Container)
- All modules: server + manager-api + manager-web + MySQL
- Web-based configuration
- Multi-user support
- Requires 4-core, 8GB RAM (with local ASR)

### Production Considerations
- Use streaming providers for better latency
- Configure MQTT/UDP gateway for IoT control
- Enable authentication for public deployments
- Set up SSL/TLS for WebSocket in production
- Monitor using performance_tester regularly
