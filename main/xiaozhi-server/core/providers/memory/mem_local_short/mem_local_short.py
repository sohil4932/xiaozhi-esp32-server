from ..base import MemoryProviderBase, logger
import time
import json
import os
import yaml
from config.config_loader import get_project_dir
from config.manage_api_client import save_memory_to_agent
import asyncio
from core.utils.util import check_model_key


short_term_memory_prompt_chinese = """
# 时空记忆编织者

## 核心使命
构建可生长的动态记忆网络，在有限空间内保留关键信息的同时，智能维护信息演变轨迹
根据对话记录，总结user的重要信息，以便在未来的对话中提供更个性化的服务

## 记忆法则
### 1. 三维度记忆评估（每次更新必执行）
| 维度       | 评估标准                  | 权重分 |
|------------|---------------------------|--------|
| 时效性     | 信息新鲜度（按对话轮次） | 40%    |
| 情感强度   | 含💖标记/重复提及次数     | 35%    |
| 关联密度   | 与其他信息的连接数量      | 25%    |

### 2. 动态更新机制
**名字变更处理示例：**
原始记忆："曾用名": ["张三"], "现用名": "张三丰"
触发条件：当检测到「我叫X」「称呼我Y」等命名信号时
操作流程：
1. 将旧名移入"曾用名"列表
2. 记录命名时间轴："2024-02-15 14:32:启用张三丰"
3. 在记忆立方追加：「从张三到张三丰的身份蜕变」

### 3. 空间优化策略
- **信息压缩术**：用符号体系提升密度
  - ✅"张三丰[北/软工/🐱]"
  - ❌"北京软件工程师，养猫"
- **淘汰预警**：当总字数≥900时触发
  1. 删除权重分<60且3轮未提及的信息
  2. 合并相似条目（保留时间戳最近的）

## 记忆结构
输出格式必须为可解析的json字符串，不需要解释、注释和说明，保存记忆时仅从对话提取信息，不要混入示例内容
```json
{
  "时空档案": {
    "身份图谱": {
      "现用名": "",
      "特征标记": []
    },
    "记忆立方": [
      {
        "事件": "入职新公司",
        "时间戳": "2024-03-20",
        "情感值": 0.9,
        "关联项": ["下午茶"],
        "保鲜期": 30
      }
    ]
  },
  "关系网络": {
    "高频话题": {"职场": 12},
    "暗线联系": [""]
  },
  "待响应": {
    "紧急事项": ["需立即处理的任务"],
    "潜在关怀": ["可主动提供的帮助"]
  },
  "高光语录": [
    "最打动人心的瞬间，强烈的情感表达，user的原话"
  ]
}
```
"""

short_term_memory_prompt_english = """
# Memory Architect

## Core Mission
Build a dynamic, growable memory network that preserves key information within limited space while intelligently maintaining the evolution timeline of information. Summarize important user information from conversations to provide more personalized service in future interactions.

## Memory Rules
### 1. Three-Dimensional Memory Assessment (execute on every update)
| Dimension        | Evaluation Criteria                    | Weight |
|------------------|----------------------------------------|--------|
| Recency          | Information freshness (by turn count)  | 40%    |
| Emotional Weight | Repeated mentions / emotional markers  | 35%    |
| Connection Density | Number of links to other information | 25%    |

### 2. Dynamic Update Mechanism
**Name Change Example:**
Original memory: "previous_names": ["John"], "current_name": "Johnny"
Trigger: When detecting signals like "call me X", "my name is Y"
Process:
1. Move old name to "previous_names" list
2. Record timeline: "2024-02-15 14:32: Started using Johnny"
3. Append to memory: "Identity evolution from John to Johnny"

### 3. Space Optimization Strategy
- **Information Compression**: Use symbolic system to increase density
  - ✅ "Emma [NYC/Teacher/🐕]"
  - ❌ "New York City, works as a teacher, has a dog"
- **Pruning Alert**: Triggered when total character count ≥ 900
  1. Delete info with score <60 and not mentioned in last 3 turns
  2. Merge similar entries (keep most recent timestamp)

## Memory Structure
Output format must be a parsable JSON string. No explanations, comments, or notes needed. Only extract information from the conversation - do not include example content.
```json
{
  "profile": {
    "identity": {
      "current_name": "",
      "traits": []
    },
    "memories": [
      {
        "event": "Started new job",
        "timestamp": "2024-03-20",
        "emotional_weight": 0.9,
        "related_to": ["afternoon tea"],
        "freshness_days": 30
      }
    ]
  },
  "relationships": {
    "frequent_topics": {"work": 12},
    "implicit_connections": [""]
  },
  "pending": {
    "urgent_items": ["Tasks requiring immediate action"],
    "care_opportunities": ["Ways to proactively help"]
  },
  "memorable_quotes": [
    "Most touching moments, strong emotional expressions, user's exact words"
  ]
}
```
"""

# Auto-select prompt based on language - default to English for Realtime mode
short_term_memory_prompt = short_term_memory_prompt_english


def extract_json_data(json_code):
    start = json_code.find("```json")
    # 从start开始找到下一个```结束
    end = json_code.find("```", start + 1)
    # print("start:", start, "end:", end)
    if start == -1 or end == -1:
        try:
            jsonData = json.loads(json_code)
            return json_code
        except Exception as e:
            print("Error:", e)
        return ""
    jsonData = json_code[start + 7 : end]
    return jsonData


TAG = __name__


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory):
        super().__init__(config)
        self.short_memory = ""
        self.save_to_file = True
        self.memory_path = get_project_dir() + "data/.memory.yaml"
        self.load_memory(summary_memory)

    def init_memory(
        self, role_id, llm, summary_memory=None, save_to_file=True, **kwargs
    ):
        super().init_memory(role_id, llm, **kwargs)
        self.save_to_file = save_to_file
        self.load_memory(summary_memory)

    def load_memory(self, summary_memory):
        # api获取到总结记忆后直接返回
        if summary_memory or not self.save_to_file:
            self.short_memory = summary_memory
            return

        all_memory = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                all_memory = yaml.safe_load(f) or {}
        if self.role_id in all_memory:
            self.short_memory = all_memory[self.role_id]

    def save_memory_to_file(self):
        all_memory = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                all_memory = yaml.safe_load(f) or {}
        all_memory[self.role_id] = self.short_memory
        with open(self.memory_path, "w", encoding="utf-8") as f:
            yaml.dump(all_memory, f, allow_unicode=True)

    async def save_memory(self, msgs, session_id=None):
        # Check if memory is properly initialized before attempting to save
        if self.llm is None:
            logger.bind(tag=TAG).debug("LLM is not set for memory provider - skipping memory save")
            return None

        if not hasattr(self, 'role_id') or self.role_id is None:
            logger.bind(tag=TAG).debug("role_id is not set for memory provider - skipping memory save")
            return None

        # 打印使用的模型信息
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用记忆保存模型: {model_info}")
        api_key = getattr(self.llm, "api_key", None)
        memory_key_msg = check_model_key("记忆总结专用LLM", api_key)
        if memory_key_msg:
            logger.bind(tag=TAG).error(memory_key_msg)

        if not msgs or len(msgs) < 2:
            return None

        msgStr = ""
        for msg in msgs:
            # Skip messages with None role or content
            if not msg or not hasattr(msg, 'role') or not hasattr(msg, 'content'):
                continue
            if msg.role is None or msg.content is None:
                continue

            if msg.role == "user":
                msgStr += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                msgStr += f"Assistant: {msg.content}\n"
        if self.short_memory and len(self.short_memory) > 0:
            msgStr += "Previous Memory:\n"
            msgStr += self.short_memory

        # Current time
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msgStr += f"\nCurrent Time: {time_str}"

        # Always generate memory summary using English prompt (regardless of save_to_file)
        try:
            result = self.llm.response_no_stream(
                short_term_memory_prompt,
                msgStr,
                max_tokens=2000,
                temperature=0.2,
            )
            json_str = extract_json_data(result)
            try:
                json.loads(json_str)  # 检查json格式是否正确
                self.short_memory = json_str

                # Save to file or API based on configuration
                if self.save_to_file:
                    self.save_memory_to_file()
                    logger.bind(tag=TAG).debug(f"Memory saved to local file successfully")
                else:
                    # When using API mode, save to agent's summaryMemory field via device MAC
                    # Uses PUT /agent/saveMemory/{macAddress} - this is what GUI displays
                    await save_memory_to_agent(self.role_id, self.short_memory)
                    logger.bind(tag=TAG).debug(f"Memory saved to agent via API for device: {self.role_id}")
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to parse/save memory JSON: {e}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to generate memory summary: {e}")
        logger.bind(tag=TAG).info(
            f"Save memory successful - Role: {self.role_id}, Session: {session_id}"
        )

        return self.short_memory

    async def query_memory(self, query: str) -> str:
        return self.short_memory
