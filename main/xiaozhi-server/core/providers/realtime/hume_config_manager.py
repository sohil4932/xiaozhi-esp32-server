"""Hume.ai EVI Configuration Manager

This module manages Hume EVI configurations programmatically via REST API,
allowing dynamic creation and updating of configs with tools/functions.

This eliminates the need to manually configure tools in the Hume web console.
"""

import json
import uuid
import aiohttp
from typing import Dict, Any, List, Optional
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class HumeConfigManager:
    """Manages Hume EVI configurations via REST API"""

    BASE_URL = "https://api.hume.ai/v0/evi"

    def __init__(self, api_key: str):
        """Initialize Hume Config Manager

        Args:
            api_key: Hume API key for authentication
        """
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-Hume-Api-Key": api_key
        }

    async def create_config_with_tools(
        self,
        name: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        language_model: str = "HUME_AI",
        model_resource: str = "evi-4-mini",
        evi_version: str = "4"
    ) -> Optional[str]:
        """Create a new Hume EVI config with tools

        Args:
            name: Config name
            tools: List of tool definitions from UnifiedToolHandler
            system_prompt: Optional system prompt/instructions
            language_model: LLM provider (HUME_AI for EVI 4-mini, or ANTHROPIC, OPENAI, GOOGLE for EVI 3)
            model_resource: Specific model name (evi-4-mini for EVI 4, or claude-3-5-sonnet-latest for EVI 3)
            evi_version: EVI version (default: "4" for EVI 4-mini)

        Returns:
            config_id if successful, None otherwise
        """
        try:
            # Convert xiaozhi tools to Hume tool format
            hume_tools = self._convert_tools_to_hume_format(tools)

            # Build request payload
            payload = {
                "name": name,
                "evi_version": evi_version,
                "tools": hume_tools,
                "language_model": {
                    "model_provider": language_model,
                    "model_resource": model_resource
                }
            }

            # Add system prompt if provided
            if system_prompt:
                payload["system_prompt"] = {
                    "text": system_prompt
                }

            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/configs",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        config_id = result.get("id")
                        logger.bind(tag=TAG).info(
                            f"Created Hume config: {name} | ID: {config_id} | Tools: {len(hume_tools)}"
                        )
                        return config_id
                    else:
                        error_text = await response.text()
                        logger.bind(tag=TAG).error(
                            f"Failed to create Hume config: {response.status} - {error_text}"
                        )
                        return None

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error creating Hume config: {e}")
            return None

    async def update_config_tools(
        self,
        config_id: str,
        tools: List[Dict[str, Any]],
        language_model: str = "HUME_AI",
        model_resource: str = "evi-4-mini",
        evi_version: str = "4"
    ) -> bool:
        """Update an existing config with new tools (creates new version)

        Args:
            config_id: Existing config ID
            tools: Updated list of tool definitions
            language_model: LLM provider
            model_resource: Specific model name
            evi_version: EVI version

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert tools to Hume format
            hume_tools = self._convert_tools_to_hume_format(tools)

            # Build request payload
            payload = {
                "evi_version": evi_version,
                "tools": hume_tools,
                "language_model": {
                    "model_provider": language_model,
                    "model_resource": model_resource
                }
            }

            # Make API request (creates new version)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/configs/{config_id}",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.bind(tag=TAG).info(
                            f"Updated Hume config: {config_id} | Tools: {len(hume_tools)}"
                        )
                        return True
                    else:
                        error_text = await response.text()
                        logger.bind(tag=TAG).error(
                            f"Failed to update Hume config: {response.status} - {error_text}"
                        )
                        return False

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error updating Hume config: {e}")
            return False

    async def list_configs(self) -> List[Dict[str, Any]]:
        """List all EVI configs

        Returns:
            List of config objects
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/configs",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        configs = result.get("configs_page", [])
                        logger.bind(tag=TAG).info(f"Found {len(configs)} Hume configs")
                        return configs
                    else:
                        logger.bind(tag=TAG).error(f"Failed to list configs: {response.status}")
                        return []

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error listing configs: {e}")
            return []

    async def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific config

        Args:
            config_id: Config ID

        Returns:
            Config object or None
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/configs/{config_id}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        config = await response.json()
                        logger.bind(tag=TAG).info(f"Retrieved Hume config: {config_id}")
                        return config
                    else:
                        logger.bind(tag=TAG).error(f"Failed to get config: {response.status}")
                        return None

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error getting config: {e}")
            return None

    def _convert_tools_to_hume_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert xiaozhi tool definitions to Hume EVI format

        Args:
            tools: List of tool definitions from UnifiedToolHandler
                Format: [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

        Returns:
            List of Hume-formatted tools
        """
        hume_tools = []

        for tool in tools:
            if tool.get("type") != "function":
                continue

            func_info = tool.get("function", {})
            name = func_info.get("name")
            description = func_info.get("description", "")
            parameters = func_info.get("parameters", {})

            if not name:
                logger.bind(tag=TAG).warning(f"Skipping tool without name: {func_info}")
                continue

            # Create Hume tool definition
            hume_tool = {
                "name": name,
                "description": description,
                "parameters": json.dumps(parameters),  # Hume expects stringified JSON
                "tool_type": "FUNCTION",
                # Generate consistent UUID from tool name for idempotency
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"xiaozhi.{name}")),
                "version": 1
            }

            hume_tools.append(hume_tool)
            logger.bind(tag=TAG).debug(f"Converted tool: {name}")

        return hume_tools


async def auto_create_config_if_needed(
    api_key: str,
    config_id: Optional[str],
    tools: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    config_name: str = "xiaozhi-auto-config"
) -> Optional[str]:
    """Automatically create or update Hume config with current tools

    This function checks if a config_id is provided:
    - If yes: Updates it with current tools
    - If no: Creates a new config with current tools

    Args:
        api_key: Hume API key
        config_id: Optional existing config ID
        tools: Current tool definitions
        system_prompt: Optional system prompt
        config_name: Name for new config if creating

    Returns:
        config_id to use (existing or newly created)
    """
    try:
        manager = HumeConfigManager(api_key)

        # If config_id provided, update it
        if config_id:
            logger.bind(tag=TAG).info(f"Updating existing Hume config: {config_id}")
            success = await manager.update_config_tools(config_id, tools)
            if success:
                return config_id
            else:
                logger.bind(tag=TAG).warning(f"Failed to update config, will create new one")

        # Create new config
        logger.bind(tag=TAG).info(f"Creating new Hume config with {len(tools)} tools")
        new_config_id = await manager.create_config_with_tools(
            name=config_name,
            tools=tools,
            system_prompt=system_prompt
        )

        if new_config_id:
            logger.bind(tag=TAG).success(
                f"Auto-created Hume config: {new_config_id} | "
                f"Add to config.yaml: config_id: \"{new_config_id}\""
            )
            return new_config_id
        else:
            logger.bind(tag=TAG).error("Failed to auto-create Hume config")
            return None

    except Exception as e:
        logger.bind(tag=TAG).error(f"Error in auto_create_config_if_needed: {e}")
        return None
