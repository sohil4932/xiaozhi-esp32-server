import httpx
import openai
from openai.types import CompletionUsage
from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    """Sarvam AI LLM Provider (Sarvam-M)

    Uses Sarvam AI's 24B parameter multilingual language model.
    Supports 10 Indic languages with native cultural context.
    Compatible with OpenAI API format.
    """

    def __init__(self, config):
        self.model_name = config.get("model_name", "sarvam-m")
        self.api_key = config.get("api_key")

        # Sarvam AI base URL
        self.base_url = config.get("base_url", "https://api.sarvam.ai/v1")

        timeout = config.get("timeout", 300)
        self.timeout = int(timeout) if timeout else 300

        # Sarvam-specific parameters
        param_defaults = {
            "max_tokens": int,
            "temperature": lambda x: round(float(x), 1),
            "top_p": lambda x: round(float(x), 1),
        }

        for param, converter in param_defaults.items():
            value = config.get(param)
            try:
                setattr(
                    self,
                    param,
                    converter(value) if value not in (None, "") else None,
                )
            except (ValueError, TypeError):
                setattr(self, param, None)

        # Sarvam-specific features
        self.reasoning_effort = config.get("reasoning_effort", None)  # low, medium, high
        self.wiki_grounding = config.get("wiki_grounding", False)

        logger.debug(
            f"Sarvam AI LLM初始化: model={self.model_name}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, top_p={self.top_p}, reasoning_effort={self.reasoning_effort}"
        )

        model_key_msg = check_model_key("LLM", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

        # Use OpenAI client with Sarvam base URL (API compatible)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            default_headers={"api-subscription-key": self.api_key}
        )

    @staticmethod
    def normalize_dialogue(dialogue):
        """自动修复 dialogue 中缺失 content 的消息"""
        for msg in dialogue:
            if "role" in msg and "content" not in msg:
                msg["content"] = ""
        return dialogue

    def response(self, session_id, dialogue, **kwargs):
        try:
            logger.bind(tag=TAG).info(f"🚀 Sarvam AI LLM调用开始 | session: {session_id} | model: {self.model_name}")

            dialogue = self.normalize_dialogue(dialogue)

            # Log the last user message for debugging
            if dialogue and len(dialogue) > 0:
                last_msg = dialogue[-1]
                logger.bind(tag=TAG).info(f"📝 用户输入: {last_msg.get('content', '')[:100]}...")

            request_params = {
                "model": self.model_name,
                "messages": dialogue,
                "stream": True,
            }

            # 添加可选参数,只有当参数不为None时才添加
            optional_params = {
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
            }

            for key, value in optional_params.items():
                if value is not None:
                    request_params[key] = value

            # Add Sarvam-specific parameters
            if self.reasoning_effort:
                request_params["reasoning_effort"] = self.reasoning_effort
            if self.wiki_grounding:
                request_params["wiki_grounding"] = self.wiki_grounding

            logger.bind(tag=TAG).info(f"📤 Sarvam AI请求参数: {request_params}")

            responses = self.client.chat.completions.create(**request_params)
            logger.bind(tag=TAG).info("✅ Sarvam API连接成功，开始接收流式响应")

            is_active = True
            response_started = False
            for chunk in responses:
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    content = getattr(delta, "content", "") if delta else ""
                except IndexError:
                    content = ""
                if content:
                    if not response_started:
                        logger.bind(tag=TAG).info(f"📨 首次响应内容: {content[:50]}...")
                        response_started = True

                    # Handle thinking mode (similar to OpenAI)
                    if "<think>" in content:
                        is_active = False
                        content = content.split("<think>")[0]
                    if "</think>" in content:
                        is_active = True
                        content = content.split("</think>")[-1]
                    if is_active:
                        yield content

            logger.bind(tag=TAG).info("✅ Sarvam AI响应完成")

        except Exception as e:
            logger.bind(tag=TAG).error(f"❌ Sarvam AI响应生成错误: {e}")
            logger.bind(tag=TAG).error(f"错误详情: {type(e).__name__}: {str(e)}")
            yield f"【Sarvam AI服务响应异常: {e}】"

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        """
        Sarvam AI function calling support
        Note: Check if Sarvam-M supports function calling, fallback to regular response if not
        """
        try:
            dialogue = self.normalize_dialogue(dialogue)

            request_params = {
                "model": self.model_name,
                "messages": dialogue,
                "stream": True,
            }

            # Try to add tools if supported
            if functions:
                request_params["tools"] = functions

            optional_params = {
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
            }

            for key, value in optional_params.items():
                if value is not None:
                    request_params[key] = value

            # Add Sarvam-specific parameters
            if self.reasoning_effort:
                request_params["reasoning_effort"] = self.reasoning_effort
            if self.wiki_grounding:
                request_params["wiki_grounding"] = self.wiki_grounding

            stream = self.client.chat.completions.create(**request_params)

            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "")
                    tool_calls = getattr(delta, "tool_calls", None)
                    yield content, tool_calls
                elif isinstance(getattr(chunk, "usage", None), CompletionUsage):
                    usage_info = getattr(chunk, "usage", None)
                    logger.bind(tag=TAG).info(
                        f"Token 消耗：输入 {getattr(usage_info, 'prompt_tokens', '未知')}，"
                        f"输出 {getattr(usage_info, 'completion_tokens', '未知')}，"
                        f"共计 {getattr(usage_info, 'total_tokens', '未知')}"
                    )

        except Exception as e:
            logger.bind(tag=TAG).error(f"Sarvam AI函数调用流式错误: {e}")
            # Fallback to regular response if function calling not supported
            logger.bind(tag=TAG).warning("尝试降级到常规响应模式")
            for content in self.response(session_id, dialogue, **kwargs):
                yield content, None
