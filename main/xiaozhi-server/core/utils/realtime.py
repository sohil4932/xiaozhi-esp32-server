"""Realtime Provider Initialization Module"""

from config.logger import setup_logging
from importlib import import_module

TAG = __name__
logger = setup_logging()


def create_instance(provider_type: str, config: dict, conn):
    """Create a Realtime provider instance

    Args:
        provider_type: Provider type (e.g., 'openai_realtime')
        config: Provider configuration
        conn: Connection handler instance

    Returns:
        Realtime provider instance
    """
    provider_map = {
        "openai_realtime": ("core.providers.realtime.openai_realtime", "OpenAIRealtimeProvider"),
        "gemini_live": ("core.providers.realtime.gemini_live", "GeminiLiveProvider"),
        "hume_realtime": ("core.providers.realtime.hume_realtime", "HumeRealtimeProvider"),
    }

    if provider_type not in provider_map:
        raise ValueError(f"Unsupported Realtime provider type: {provider_type}")

    try:
        module_path, class_name = provider_map[provider_type]
        module = import_module(module_path)

        # Get the provider class
        provider_class = getattr(module, class_name)

        # Instantiate and return
        provider = provider_class(config, conn)
        logger.bind(tag=TAG).info(f"Realtime provider created: {provider_type}")
        return provider

    except Exception as e:
        logger.bind(tag=TAG).error(f"Failed to create Realtime provider {provider_type}: {e}")
        raise


def initialize_realtime(config: dict, conn):
    """Initialize Realtime provider based on configuration

    Args:
        config: Full configuration dictionary
        conn: Connection handler instance

    Returns:
        Realtime provider instance or None if not configured
    """
    try:
        # Check if Realtime mode is selected
        selected_realtime = config.get("selected_module", {}).get("Realtime")
        if not selected_realtime:
            return None

        # Get provider configuration
        realtime_config = config.get("Realtime", {}).get(selected_realtime)
        if not realtime_config:
            logger.bind(tag=TAG).warning(f"Realtime provider {selected_realtime} not found in config")
            return None

        # Get provider type
        provider_type = realtime_config.get("type")
        if not provider_type:
            logger.bind(tag=TAG).warning(f"Realtime provider {selected_realtime} missing 'type' field")
            return None

        # Create provider instance
        return create_instance(provider_type, realtime_config, conn)

    except Exception as e:
        logger.bind(tag=TAG).error(f"Failed to initialize Realtime provider: {e}")
        return None
