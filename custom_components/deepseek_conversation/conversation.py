    """Conversation support for DeepSeek."""

    from collections.abc import AsyncGenerator, Callable
    import json
    # Removed Literal import as it might not be strictly needed now
    from typing import Any, AsyncGenerator, Callable, Literal, cast, Optional, Union, Dict, List

    import openai
    # Import necessary types for chat completions
    from openai import AsyncStream
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    # --- Removed problematic import ---
    # from openai.types.chat.completion_create_params import ToolChoiceParam # For tool choice if needed
    # --- End removal ---

    # Keep voluptuous_openapi if tool schemas are complex, otherwise remove
    # from voluptuous_openapi import convert
    import voluptuous as vol # Keep for basic schema validation if needed

    from homeassistant.components import assist_pipeline, conversation
    from homeassistant.config_entries import ConfigEntry
    # --- Import CONF_LLM_HASS_API ---
    from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API
    # --- End Import ---
    from homeassistant.core import HomeAssistant
    # --- Import HomeAssistantError --- Needed for exception handling
    from homeassistant.exceptions import HomeAssistantError
    # --- End Import ---
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
        LOGGER, # Use the logger from const
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
    ) -> Dict[str, Any]: # Changed return type hint to generic Dict
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
            role: Optional[Literal["user", "assistant", "tool"]] = None # Made Optional explicit
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
                    # Ensure tool_calls are correctly formatted if present
                    formatted_tool_calls = []
                    for tc in content.tool_calls:
                        # Ensure tool_args is a string, as expected by the API usually
                        arguments_str = json.dumps(tc.tool_args) if not isinstance(tc.tool_args, str) else tc.tool_args
                        formatted_tool_calls.append(
                             ChatCompletionMessageToolCall(
                                id=tc.id,
                                function=dict(name=tc.tool_name, arguments=arguments_str),
                                type="function"
                            )
                        )
                    tool_calls = formatted_tool_calls
            elif isinstance(content, conversation.ToolResultContent):
                role = "tool"
                message_content = json.dumps(content.tool_result) # Tool results are content for 'tool' role
                tool_call_id = content.tool_call_id

            # Construct the message dictionary
            if role:
                msg: Dict[str, Any] = {"role": role} # Use Dict for type hint
                if message_content:
                    msg["content"] = message_content
                if tool_calls:
                     # Ensure content is None or empty string if tool_calls are present for assistant
                    if role == "assistant":
                        msg["content"] = msg.get("content") # None if no text content
                    # Use model_dump for pydantic models if needed, or direct dict if already formatted
                    msg["tool_calls"] = [tc.model_dump(exclude_unset=True) if hasattr(tc, 'model_dump') else tc for tc in tool_calls]
                if tool_call_id:
                    msg["tool_call_id"] = tool_call_id

                messages.append(msg)

        return messages
    # --- End Message Conversion ---


    # --- Stream Transformation (Adapted for ChatCompletionChunk) ---
    async def _transform_stream(
        chat_log: conversation.ChatLog,
        result: AsyncStream[ChatCompletionChunk],
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]: # Added None to AsyncGenerator
        """Transform a DeepSeek delta stream (ChatCompletionChunk) into HA format."""
        current_tool_calls: list[dict] = []
        current_tool_call_args_buffer: dict[int, str] = {} # Store partial args per index
        role: Optional[Literal["assistant"]] = None # Made Optional explicit
        full_response_log = [] # --- DEBUG: Log full stream ---

        async for chunk in result:
            # --- DEBUG: Log each chunk ---
            LOGGER.debug("DeepSeek Stream Chunk: %s", chunk.model_dump_json(indent=2))
            full_response_log.append(chunk.model_dump())
            # --- END DEBUG ---

            delta = chunk.choices[0].delta if chunk.choices else None
            finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

            if not delta:
                continue

            # Role appears only in the first chunk
            if delta.role:
                # Ensure role is 'assistant' as expected
                if delta.role == "assistant":
                    role = delta.role
                    yield {"role": role}
                else:
                    LOGGER.warning("Unexpected role in stream delta: %s", delta.role)


            # Handle content delta
            if delta.content:
                yield {"content": delta.content}

            # Handle tool call deltas (more complex)
            if delta.tool_calls:
                # --- DEBUG: Log tool call chunk ---
                LOGGER.debug("Received Tool Call Chunk: %s", delta.tool_calls)
                # --- END DEBUG ---
                for tool_call_chunk in delta.tool_calls:
                    # Ensure index is not None before proceeding
                    if tool_call_chunk.index is None:
                        LOGGER.warning("Tool call chunk missing index: %s", tool_call_chunk)
                        continue

                    index = tool_call_chunk.index

                    # New tool call started
                    if index >= len(current_tool_calls):
                        # Ensure list is long enough
                        current_tool_calls.extend([{}] * (index - len(current_tool_calls) + 1))
                        # Store initial tool call info (id, type, function name)
                        # Check if function exists and has name before accessing
                        function_name = tool_call_chunk.function.name if tool_call_chunk.function else None
                        if tool_call_chunk.id and tool_call_chunk.type and function_name:
                            current_tool_calls[index] = {
                                "id": tool_call_chunk.id,
                                "type": tool_call_chunk.type,
                                "function": {"name": function_name, "arguments": ""}
                            }
                            current_tool_call_args_buffer[index] = "" # Initialize buffer for this index
                            # --- DEBUG: Log new tool call start ---
                            LOGGER.debug("Tool Call Start Detected: Index=%d, ID=%s, Name=%s", index, tool_call_chunk.id, function_name)
                            # --- END DEBUG ---
                        else:
                             LOGGER.warning("Incomplete tool call start info in chunk: %s", tool_call_chunk)


                    # Append argument delta to the buffer
                    # Ensure function and arguments exist before appending
                    if tool_call_chunk.function and tool_call_chunk.function.arguments and index in current_tool_call_args_buffer:
                        current_tool_call_args_buffer[index] += tool_call_chunk.function.arguments
                        # --- DEBUG: Log argument delta ---
                        # LOGGER.debug("Tool Call Arg Delta: Index=%d, Args=%s", index, tool_call_chunk.function.arguments) # Can be noisy
                        # --- END DEBUG ---


            # Check finish reason
            if finish_reason:
                 # --- DEBUG: Log finish reason and final tool state ---
                LOGGER.debug("Stream Finish Reason: %s", finish_reason)
                LOGGER.debug("Final Tool Args Buffer: %s", current_tool_call_args_buffer)
                LOGGER.debug("Final Current Tool Calls: %s", current_tool_calls)
                # --- END DEBUG ---
                if finish_reason == "tool_calls":
                    # Process completed tool calls
                    tool_inputs = []
                    for index, args_str in current_tool_call_args_buffer.items():
                        if index < len(current_tool_calls) and current_tool_calls[index]: # Check if entry exists
                            tool_call_info = current_tool_calls[index]
                            # Ensure function info exists
                            if "function" in tool_call_info and "name" in tool_call_info["function"]:
                                try:
                                    # Parse arguments once fully received
                                    # --- DEBUG: Log raw args before parsing ---
                                    LOGGER.debug("Attempting to parse args for %s: %s", tool_call_info["function"]["name"], args_str)
                                    # --- END DEBUG ---
                                    tool_args = json.loads(args_str) if args_str else {} # Handle empty args
                                    tool_inputs.append(
                                        llm.ToolInput(
                                            id=tool_call_info["id"],
                                            tool_name=tool_call_info["function"]["name"],
                                            tool_args=tool_args,
                                        )
                                    )
                                    # --- DEBUG: Log successful tool input ---
                                    LOGGER.debug("Successfully parsed tool input: %s", tool_inputs[-1])
                                    # --- END DEBUG ---
                                except json.JSONDecodeError as e:
                                    LOGGER.error(
                                        "Failed to decode tool arguments for %s: %s. Error: %s",
                                        tool_call_info["function"]["name"], args_str, e
                                    )
                            else:
                                 LOGGER.warning("Missing function info for tool call at index %d", index)
                    if tool_inputs:
                        yield {"tool_calls": tool_inputs}
                    # Clear buffers for next iteration if any
                    current_tool_calls = []
                    current_tool_call_args_buffer = {}

                elif finish_reason == "stop":
                    # Conversation finished normally
                    pass # Nothing specific to yield here
                elif finish_reason == "length":
                    # --- Raise specific error for length limit ---
                    raise HomeAssistantError("max_token") # Use a simple key
                    # --- End Raise ---
                elif finish_reason == "content_filter":
                     raise HomeAssistantError("content_filter") # Use a simple key
                else:
                     raise HomeAssistantError(f"finish_reason_{finish_reason}") # Use finish reason as key

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
            # --- Re-enable supported_features based on options ---
            # Check if CONF_LLM_HASS_API exists and is not 'none' or None
            if self.entry.options.get(CONF_LLM_HASS_API):
                self._attr_supported_features = (
                    conversation.ConversationEntityFeature.CONTROL
                )
            # --- End re-enable ---

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
            # Ensure runtime_data exists before accessing client
            if not hasattr(self.entry, 'runtime_data') or not isinstance(self.entry.runtime_data, openai.AsyncClient):
                 LOGGER.error("DeepSeek client not available in runtime_data.")
                 # --- Use intent response for error ---
                 intent_response = intent.IntentResponse(language=user_input.language)
                 intent_response.async_set_error(
                      intent.IntentResponseErrorCode.UNKNOWN,
                      "DeepSeek client not available" # Keep error message internal
                 )
                 return conversation.ConversationResult(
                     response=intent_response, conversation_id=chat_log.conversation_id
                 )
                 # --- End error handling ---
            client: openai.AsyncClient = self.entry.runtime_data
            model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

            # Get system prompt from options
            system_prompt = options.get(CONF_PROMPT)

            # --- Prepare tools if HASS API is used ---
            tools: list[Dict[str, Any]] | None = None # Use generic Dict hint
            tool_choice: Optional[Union[str, Dict[str, Any]]] = None # Use generic Dict/str hint

            # --- Get selected HASS API from options ---
            # --- MODIFIED LOGIC: Use chat_log.llm_api directly ---
            if chat_log.llm_api:
                # The llm_api object provided in chat_log already has the correct context
                active_llm_api = chat_log.llm_api
                tools = [
                    _format_tool(tool, active_llm_api.custom_serializer)
                    for tool in active_llm_api.tools
                ]
                tool_choice = "auto" # Let the LLM decide when to use tools
                # --- DEBUG: Log tools being sent ---
                LOGGER.debug("Sending tools to DeepSeek: %s", json.dumps(tools, indent=2))
                # --- END DEBUG ---
            # --- End MODIFIED LOGIC ---


            # Removed web search tool logic

            # Convert chat history to API format
            messages = _convert_content_to_messages(chat_log.content, system_prompt)
            # --- DEBUG: Log messages being sent ---
            LOGGER.debug("Sending messages to DeepSeek: %s", json.dumps(messages, indent=2))
            # --- END DEBUG ---

            # To prevent infinite loops with tools
            for _iteration in range(MAX_TOOL_ITERATIONS):
                model_args: Dict[str, Any] = { # Use Dict for type hint
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

                # --- DEBUG: Log model args ---
                LOGGER.debug("Model arguments for DeepSeek: %s", model_args)
                # --- END DEBUG ---

                try:
                    result = await client.chat.completions.create(**model_args)
                except openai.RateLimitError as err:
                    LOGGER.warning("Rate limited by DeepSeek: %s", err)
                    # --- Use intent response for error ---
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN, # Or map to a specific code if possible
                        "Rate limited by DeepSeek API"
                    )
                    return conversation.ConversationResult(
                        response=intent_response, conversation_id=chat_log.conversation_id
                    )
                    # --- End error handling ---
                except openai.APIConnectionError as err:
                     LOGGER.error("Connection error talking to DeepSeek: %s", err)
                     # --- Use intent response for error ---
                     intent_response = intent.IntentResponse(language=user_input.language)
                     intent_response.async_set_error(
                         intent.IntentResponseErrorCode.UNKNOWN, # Or map to a specific code if possible
                         "Connection error with DeepSeek API"
                     )
                     return conversation.ConversationResult(
                         response=intent_response, conversation_id=chat_log.conversation_id
                     )
                    # --- End error handling ---
                except openai.OpenAIError as err:
                    LOGGER.error("Error talking to DeepSeek: %s", err)
                    # --- Use intent response for error ---
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN, # Or map to a specific code if possible
                        f"DeepSeek API error: {err}"
                    )
                    return conversation.ConversationResult(
                        response=intent_response, conversation_id=chat_log.conversation_id
                    )
                    # --- End error handling ---


                # Process the stream and update chat log
                try:
                    async for content_delta in chat_log.async_add_delta_content_stream(
                        user_input.agent_id, _transform_stream(chat_log, result)
                    ):
                        # Update message history for potential next iteration (tool use)
                        # This part requires careful handling based on how chat_log structures deltas
                        # For simplicity, we rebuild messages from the updated chat_log before the next loop
                        pass # Message history will be rebuilt from chat_log below
                except HomeAssistantError as e:
                     # Catch errors raised by _transform_stream (e.g., finish reasons)
                     LOGGER.error("Error processing DeepSeek stream: %s", e)
                     # --- Use intent response for error ---
                     intent_response = intent.IntentResponse(language=user_input.language)
                     # Use specific error codes based on the simple keys raised in _transform_stream
                     error_code = intent.IntentResponseErrorCode.UNKNOWN
                     error_msg = str(e)
                     if str(e) == "max_token":
                          error_code = intent.IntentResponseErrorCode.UNKNOWN # No specific code for this
                          error_msg = "Response truncated by token limit"
                     elif str(e) == "content_filter":
                          error_code = intent.IntentResponseErrorCode.UNKNOWN # No specific code
                          error_msg = "Response blocked by content filter"

                     intent_response.async_set_error(error_code, error_msg)
                     return conversation.ConversationResult(
                         response=intent_response, conversation_id=chat_log.conversation_id
                     )
                     # --- End error handling ---


                # Rebuild messages from the potentially updated chat log for the next iteration
                messages = _convert_content_to_messages(chat_log.content, system_prompt)

                # Break loop if no more tool results need processing
                if not chat_log.unresponded_tool_results:
                    # --- DEBUG: Log successful completion or no tool call needed ---
                    LOGGER.debug("Iteration %d finished. No unresponded tool results.", _iteration + 1)
                    # --- END DEBUG ---
                    break
                else:
                     # --- DEBUG: Log tool results received, preparing next iteration ---
                     LOGGER.debug("Iteration %d finished. Unresponded tool results found, preparing next iteration.", _iteration + 1)
                     # --- END DEBUG ---
                     pass # Continue loop

            else:
                # Loop finished without break, meaning MAX_TOOL_ITERATIONS reached
                LOGGER.warning("Max tool iterations reached for conversation %s", chat_log.conversation_id)
                # --- Use intent response for error ---
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN, # No specific code for this
                    "Maximum tool iterations reached"
                 )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=chat_log.conversation_id
                )
                # --- End error handling ---


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

    
