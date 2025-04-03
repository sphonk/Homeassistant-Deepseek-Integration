from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

import aiohttp
import logging

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry):
    async def converse(hass: HomeAssistant, text: str, conversation_id: str, language: str):
        config = entry.data
        
        messages = [{
            "role": "user",
            "content": text
        }]
        
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.get("model", "deepseek-chat"),
            "messages": messages,
            "temperature": config.get("temperature", 0.5),
            "max_tokens": config.get("max_tokens", 150)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    response_text = result["choices"][0]["message"]["content"]
                    return intent.IntentResponse(
                        response_type=intent.IntentResponseType.ACTION_DONE,
                        speech={"plain": {"speech": response_text}}
                    )
                    
        except aiohttp.ClientError as err:
            _LOGGER.error("DeepSeek API error: %s", err)
            raise intent.IntentHandleError("Error communicating with DeepSeek API")
            
    conversation.async_set_conversation_api(hass, converse)
    return True
