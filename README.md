# Home Assistant DeepSeek Conversation Integration (Prototype)

> **Important Note:** This is an **experimental prototype integration**! It has been adapted to use the DeepSeek API instead of the OpenAI API. It may be unstable, contain bugs, and might not be suitable for production use. It serves as a foundation and demonstration for the potential integration of DeepSeek into Home Assistant.

## Description

This custom integration for Home Assistant enables the use of the [DeepSeek API](https://platform.deepseek.com/) as a conversation agent for Assist. It is based on the official OpenAI Conversation integration from Home Assistant but has been modified to communicate with the DeepSeek API endpoint and corresponding models (e.g., `deepseek-chat`).

## Current Status & Features

* **Conversation Agent:** Allows using DeepSeek for text-based conversations via the Assist interface.
* **UI Configuration:** API key and basic model parameters (model name, max tokens, temperature, top P, system prompt, HASS API for control) can be configured through the Home Assistant user interface.
* **Streaming:** Responses are streamed.
* **Device Control (Experimental):** The integration attempts to pass the standard Home Assistant tools (Tools/Functions) to DeepSeek if a HASS API is selected in the options. The ability to actually control devices **strongly depends on how well the used DeepSeek model understands and supports the OpenAI-compatible tool-calling mechanism.** This may be unreliable.
* **Removed Features:** The original DALL-E image generation function has been removed.

## Installation (via HACS)

1.  Ensure [HACS (Home Assistant Community Store)](https://hacs.xyz/) is installed.
2.  In HACS, go to the "Integrations" section.
3.  Click the three dots in the top right corner and select "Custom Repositories".
4.  Add the URL of this GitHub repository.
5.  Select "Integration" as the category.
6.  Click "Add".
7.  Find the "DeepSeek Conversation" integration in the list and click "Install".
8.  **Restart Home Assistant.**

## Configuration

1.  Go to "Settings" -> "Devices & Services".
2.  Click "+ Add Integration" in the bottom right corner.
3.  Search for "DeepSeek Conversation" and select it.
4.  Enter your **DeepSeek API Key** when prompted.
5.  After successful setup, you can customize the integration via "Configure":
    * **API Key:** (Already entered)
    * **Instructions (Prompt):** The system prompt for the model (can contain Jinja2 templates).
    * **LLM Home Assistant API:** Select which Home Assistant functions should be provided for device control (e.g., "Assist" for standard functions, "No control" to disable).
    * **Model:** The name of the DeepSeek model to use (Default: `deepseek-chat`).
    * **Maximum tokens:** Maximum length of the response (Default: 1500). Increase this value if needed.
    * **Temperature / Top P:** Parameters to control the creativity/randomness of the response.

## Requirements

* A valid API key from [DeepSeek](https://platform.deepseek.com/).
* HACS (for easy installation).

## Disclaimer

This is a prototype and is provided "as is". There is no guarantee of functionality or stability. Use at your own risk.

## Future Development

This code can serve as a starting point for a more stable and official DeepSeek integration. Potential improvements could include:
* Better error handling.
* Optimization of tool calls specifically for DeepSeek (if deviations from the OpenAI standard exist).
* Adding more configuration options from the DeepSeek API.
* Resolving the deprecation warnings.

