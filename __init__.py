from .config_flow import DeepSeekConfigFlow
from .const import DOMAIN

async def async_setup(hass, config):
    hass.config_entries.flow.async_register_flow(
        DOMAIN, DeepSeekConfigFlow()
    )
    return True
