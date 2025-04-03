"""Config flow for DeepSeek Conversation integration."""

from __future__ import annotations

# Removed json import as it's no longer used here
import logging
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol
# Removed voluptuous_openapi import as it's no longer used here

# Removed zone import
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY # Removed unused ATTR_* and CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm # Keep llm for API selection if needed, or remove
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    # SelectSelectorMode, # Removed unused mode
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

# Updated imports from const
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    # Removed CONF_REASONING_EFFORT, CONF_RECOMMENDED, CONF_WEB_SEARCH*
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN, # Use updated domain
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    DEEPSEEK_API_BASE_URL, # Use base URL constant
    # Removed RECOMMENDED_REASONING_EFFORT, RECOMMENDED_WEB_SEARCH*
    # Removed UNSUPPORTED_MODELS
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

# Simplified default options for DeepSeek
DEFAULT_OPTIONS = {
    # CONF_LLM_HASS_API: llm.LLM_API_ASSIST, # Keep if HASS API control is desired
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT, # Keep default prompt
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    client = openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY],
        base_url=DEEPSEEK_API_BASE_URL, # Use DeepSeek URL for validation
        http_client=get_async_client(hass)
    )
    # Validate using a test chat completion call
    await client.with_options(timeout=10.0).chat.completions.create(
        model=RECOMMENDED_CHAT_MODEL, # Use default model for test
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1,
    )


class DeepSeekConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for DeepSeek Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except openai.OpenAIError as e:
            _LOGGER.error("DeepSeek API error during validation: %s", e)
            errors["base"] = "api_error" # Add a generic API error key
        except Exception:
            _LOGGER.exception("Unexpected exception during validation")
            errors["base"] = "unknown"
        else:
            # Changed title to DeepSeek
            return self.async_create_entry(
                title="DeepSeek",
                data=user_input,
                options=DEFAULT_OPTIONS, # Use simplified default options
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return DeepSeekOptionsFlow(config_entry)


class DeepSeekOptionsFlow(OptionsFlow):
    """DeepSeek config flow options handler."""

    # Removed last_rendered_recommended logic

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry # Store config entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Basic validation (e.g., check if model name is reasonable if needed)
            # Add more validation if required for DeepSeek models/params

            # Removed logic related to CONF_RECOMMENDED toggle
            # Removed logic related to CONF_LLM_HASS_API == "none" (handle directly in schema/defaults)
            # Removed logic checking UNSUPPORTED_MODELS
            # Removed logic checking web search support and location data

            if not errors:
                # Ensure required options have values before creating entry
                updated_options = {**self.config_entry.options, **user_input}
                return self.async_create_entry(title="", data=updated_options)

        # Use the simplified schema function
        schema = deepseek_config_option_schema(self.hass, self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    # Removed get_location_data method


def deepseek_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> VolDictType:
    """Return a schema for DeepSeek completion options."""
    # Keep HASS API selection if desired, otherwise remove
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label="No control", value="none")
    ]
    hass_apis.extend(
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    )

    # Simplified schema, removed OpenAI specific options
    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT, # Ensure default
        ): TemplateSelector(),
        # Keep HASS API selector if needed
        # vol.Optional(
        #     CONF_LLM_HASS_API,
        #     description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        #     default="none",
        # ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            default=RECOMMENDED_CHAT_MODEL,
        ): str,
        vol.Optional(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            default=RECOMMENDED_MAX_TOKENS,
        ): int,
        vol.Optional(
            CONF_TOP_P,
            description={"suggested_value": options.get(CONF_TOP_P)},
            default=RECOMMENDED_TOP_P,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options.get(CONF_TEMPERATURE)},
            default=RECOMMENDED_TEMPERATURE,
        ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05, mode="slider")),
        # Removed CONF_RECOMMENDED toggle
        # Removed CONF_REASONING_EFFORT
        # Removed CONF_WEB_SEARCH options
    }
    return schema

