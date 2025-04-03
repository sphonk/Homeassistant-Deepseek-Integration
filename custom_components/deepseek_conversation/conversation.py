"""Conversation support for DeepSeek."""

from collections.abc import AsyncGenerator, Callable
import json
from typing import Any, Literal, cast

import openai
# Import necessary types for chat completions
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.completion_create_params import ToolChoiceParam # For tool choice if needed

# Removed unused OpenAI response types
# from openai.types.responses import (...)

# Keep voluptuous_openapi if tool schemas are complex, otherwise remove
# from voluptuous_openapi import convert
import voluptuous as vol # Keep for basic schema validation if needed

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
# Removed CONF_LLM_HASS_API, MATCH_ALL (can be added back if needed)
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

# Use the specific type alias if defined, otherwise generic ConfigEntry
# from . import DeepSeekConfigEntry
type DeepSeekConfigEntry = ConfigEntry

# Updated imports from const
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    # Removed CONF_REASONING_EFFORT, CONF_WEB_SEARCH*
    DOMAIN, # Use updated domain
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    # Removed RECOMMENDED_REASONING_EFFORT, RECOMMENDED_WEB_SEARCH*
)

# Max number of back and forth with the LLM for tool usage
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: DeepSeekConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    # Ensure runtime_data is initialized (should be done in __init__.py)
    if not hasattr(config_entry, 'runtime_data') or config_entry.runtime_data is None:
        LOGGER.error("DeepSeek client not initialized in config entry.")
        # Depending on HA version, setup might fail earlier, but good to check
        return

    agent = DeepSeekConversationEntity(config_entry)
    async_add_entities([agent])


# --- Tool Formatting (Keep if using tools with DeepSeek) ---
def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> llm.ToolSchema:
    """Format tool specification for OpenAI-compatible tool format."""
    # Voluptuous-openapi might be overkill if schemas are simple JSON schemas
    # parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    # Basic conversion assuming tool.parameters is already JSON schema compatible
    parameters = tool.parameters.schema if isinstance(tool.parameters, vol.Schema) else tool.parameters

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }
    }
# --- End Tool Formatting ---


# --- Message Conversion (Adapted for chat.completions) ---
def _convert_content_to_messages(
    content_list: list[conversation.Content],
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Convert conversation history to DeepSeek API message format."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for content in content_list:
        role: Literal["user", "assistant", "tool"] | None = None
        message_content: str | list[dict[str, Any]] | None = None
        tool_calls: list[ChatCompletionMessageToolCall] | None = None
        tool_call_id: str | None = None

        if isinstance(content, conversation.UserContent):
            role = "user"
            # Handle potential multi-modal content if DeepSeek supports it
            # For now, assume simple text
            message_content = content.content
        elif isinstance(content, conversation.AssistantContent):
            role = "assistant"
            message_content = content.content
            if content.tool_calls:
                tool_calls = [
                    ChatCompletionMessageToolCall(
                        id=tc.id,
                        function=dict(name=tc.tool_name, arguments=json.dumps(tc.tool_args)),
                        type="function"
                    )
                    for tc in content.tool_calls
                ]
        elif isinstance(content, conversation.ToolResultContent):
            role = "tool"
            message_content = json.dumps(content.tool_result) # Tool results are content for 'tool' role
            tool_call_id = content.tool_call_id

        # Construct the message dictionary
        if role:
            msg = {"role": role}
            if message_content:
                msg["content"] = message_content
            if tool_calls:
                 # Ensure content is None or empty string if tool_calls are present for assistant
                if role == "assistant":
                    msg["content"] = msg.get("content") or "" # Or None, check API spec
                msg["tool_calls"] = [tc.model_dump(exclude_unset=True) for tc in tool_calls]
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id

            messages.append(msg)

    return messages
# --- End Message Conversion ---


# --- Stream Transformation (Adapted for ChatCompletionChunk) ---
async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a DeepSeek delta stream (ChatCompletionChunk) into HA format."""
    current_tool_calls: list[dict] = []
    current_tool_call_args_buffer: dict[int, str] = {} # Store partial args per index
    role: Literal["assistant"] | None = None

    async for chunk in result:
        # LOGGER.debug("Received chunk: %s", chunk) # Optional: for debugging stream
        delta = chunk.choices[0].delta if chunk.choices else None
        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

        if not delta:
            continue

        # Role appears only in the first chunk
        if delta.role:
            role = delta.role
            yield {"role": role} # Should always be 'assistant'

        # Handle content delta
        if delta.content:
            yield {"content": delta.content}

        # Handle tool call deltas (more complex)
        if delta.tool_calls:
            for tool_call_chunk in delta.tool_calls:
                index = tool_call_chunk.index

                # New tool call started
                if index >= len(current_tool_calls):
                    # Ensure list is long enough
                    current_tool_calls.extend([{}] * (index - len(current_tool_calls) + 1))
                    # Store initial tool call info (id, type, function name)
                    current_tool_calls[index] = {
                        "id": tool_call_chunk.id,
                        "type": tool_call_chunk.type,
                        "function": {"name": tool_call_chunk.function.name, "arguments": ""}
                    }
                    current_tool_call_args_buffer[index] = "" # Initialize buffer for this index

                # Append argument delta to the buffer
                if tool_call_chunk.function and tool_call_chunk.function.arguments:
                    current_tool_call_args_buffer[index] += tool_call_chunk.function.arguments

        # Check finish reason
        if finish_reason:
            if finish_reason == "tool_calls":
                # Process completed tool calls
                tool_inputs = []
                for index, args_str in current_tool_call_args_buffer.items():
                    if index < len(current_tool_calls):
                        tool_call_info = current_tool_calls[index]
                        try:
                            # Parse arguments once fully received
                            tool_args = json.loads(args_str)
                            tool_inputs.append(
                                llm.ToolInput(
                                    id=tool_call_info["id"],
                                    tool_name=tool_call_info["function"]["name"],
                                    tool_args=tool_args,
                                )
                            )
                        except json.JSONDecodeError:
                            LOGGER.error(
                                "Failed to decode tool arguments for %s: %s",
                                tool_call_info["function"]["name"], args_str
                            )
                if tool_inputs:
                    yield {"tool_calls": tool_inputs}
                # Clear buffers for next iteration if any
                current_tool_calls = []
                current_tool_call_args_buffer = {}

            elif finish_reason == "stop":
                # Conversation finished normally
                pass # Nothing specific to yield here
            elif finish_reason == "length":
                raise HomeAssistantError("DeepSeek response truncated: maximum length reached.")
            elif finish_reason == "content_filter":
                 raise HomeAssistantError("DeepSeek response stopped due to content filter.")
            else:
                 raise HomeAssistantError(f"DeepSeek response stopped unexpectedly: {finish_reason}")

        # Handle usage/stats if available in the stream (less common in chunks)
        # if chunk.usage: ... (usually comes at the end or not at all in stream)

    # Final usage stats might be available after the loop if the library provides it
    # final_response = await result.get_final_response() # Check if openai lib supports this
    # if final_response and final_response.usage:
    #     chat_log.async_trace(...)

# --- End Stream Transformation ---


class DeepSeekConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """DeepSeek conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: DeepSeekConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title, # Should be "DeepSeek" now
            manufacturer="DeepSeek", # Changed manufacturer
            model="DeepSeek API",    # Changed model
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        # Re-enable if HASS API control is kept
        # if self.entry.options.get(CONF_LLM_HASS_API):
        #     self._attr_supported_features = (
        #         conversation.ConversationEntityFeature.CONTROL
        #     )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        # Migrate pipeline if needed (using the new DOMAIN)
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a message using DeepSeek."""
        options = self.entry.options
        client: openai.AsyncClient = self.entry.runtime_data
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        # Get system prompt from options
        system_prompt = options.get(CONF_PROMPT)

        # --- Prepare tools if HASS API is used ---
        tools: list[llm.ToolSchema] | None = None
        tool_choice: ToolChoiceParam | None = None # Optional: control tool usage
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
            # Example: Force tool usage if available
            # tool_choice = "auto" # or {"type": "function", "function": {"name": "specific_tool"}}

        # Removed web search tool logic

        # Convert chat history to API format
        messages = _convert_content_to_messages(chat_log.content, system_prompt)

        # To prevent infinite loops with tools
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": model,
                "messages": messages,
                "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                # "user": chat_log.conversation_id, # Optional
                "stream": True,
            }
            if tools:
                model_args["tools"] = tools
            if tool_choice:
                 model_args["tool_choice"] = tool_choice

            # Removed OpenAI specific 'reasoning' and 'store' args

            try:
                result = await client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.warning("Rate limited by DeepSeek: %s", err)
                # Return specific error type if defined in conversation
                return conversation.ConversationResult(
                     response=conversation.ConversationErrorResponse("rate_limit", DOMAIN),
                     conversation_id=chat_log.conversation_id
                )
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to DeepSeek: %s", err)
                 # Return generic error
                return conversation.ConversationResult(
                     response=conversation.ConversationErrorResponse("api_error", DOMAIN),
                     conversation_id=chat_log.conversation_id
                )

            # Process the stream and update chat log
            async for content_delta in chat_log.async_add_delta_content_stream(
                user_input.agent_id, _transform_stream(chat_log, result)
            ):
                # Update message history for potential next iteration (tool use)
                # This part requires careful handling based on how chat_log structures deltas
                # For simplicity, we rebuild messages from the updated chat_log before the next loop
                pass # Message history will be rebuilt from chat_log below

            # Rebuild messages from the potentially updated chat log for the next iteration
            messages = _convert_content_to_messages(chat_log.content, system_prompt)

            # Break loop if no more tool results need processing
            if not chat_log.unresponded_tool_results:
                break
        else:
            # Loop finished without break, meaning MAX_TOOL_ITERATIONS reached
            LOGGER.warning("Max tool iterations reached for conversation %s", chat_log.conversation_id)
            # Return an error or the last response? Returning error for now.
            return conversation.ConversationResult(
                 response=conversation.ConversationErrorResponse("max_tool_iterations", DOMAIN),
                 conversation_id=chat_log.conversation_id
            )


        # --- Construct final response ---
        intent_response = intent.IntentResponse(language=user_input.language)
        # Get the last assistant message from the log
        last_assistant_message = next(
            (msg for msg in reversed(chat_log.content) if isinstance(msg, conversation.AssistantContent)),
            None
        )
        speech_text = last_assistant_message.content if last_assistant_message else ""
        intent_response.async_set_speech(speech_text or "") # Ensure not None

        # Add tool results to the response if any were generated in the last turn
        if last_assistant_message and last_assistant_message.tool_calls:
             # This part might need adjustment based on how HA handles tool results in IntentResponse
             # For now, just setting speech. Tool results are implicitly handled by the agent flow.
             pass

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )
        # --- End final response construction ---

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload the entry to apply changes
        await hass.config_entries.async_reload(entry.entry_id)

