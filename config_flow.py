from homeassistant import config_entries
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from .const import DOMAIN

CONFIG_SCHEMA = vol.Schema({
    vol.Required("api_key"): cv.string,
    vol.Optional("model", default="deepseek-chat"): cv.string,
    vol.Optional("temperature", default=0.5): vol.Coerce(float),
    vol.Optional("max_tokens", default=150): int,
})

class DeepSeekConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    async def async_step_user(self, user_input=None):
        errors = {}
        
        if user_input is not None:
            return self.async_create_entry(title="DeepSeek Conversation", data=user_input)
            
        return self.async_show_form(
            step_id="user",
            data_schema=CONFIG_SCHEMA,
            description_placeholders={
                "api_link": "https://platform.deepseek.com/api-keys"
            },
            errors=errors
        )
