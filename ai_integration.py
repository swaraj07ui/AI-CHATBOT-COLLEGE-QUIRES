import os
from typing import Optional

try:
    import openai
except Exception:
    openai = None


class AIIntegration:
    """Lightweight wrapper for OpenAI calls used by the chatbot.

    Behavior:
    - If the `openai` package is available and `OPENAI_API_KEY` is set,
      it will call the ChatCompletion API and return the assistant text.
    - Otherwise it returns a helpful error message so the chatbot can continue
      running without crashing.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if openai and self.api_key:
            # For older openai versions this sets the global key
            try:
                openai.api_key = self.api_key
            except Exception:
                # Some distributions expect a client object instead; ignore here
                pass

    def get_openai_response(self, prompt: str, context: Optional[str] = "") -> str:
        """Return a response string for the given prompt.

        If OpenAI is not available, returns a friendly message describing the
        missing dependency or configuration.
        """
        if not openai:
            return (
                "AI service is not available because the `openai` package is not installed. "
                "Install it with `pip install openai` to enable AI responses."
            )

        if not self.api_key:
            return (
                "AI service is not configured. Please set the OPENAI_API_KEY environment "
                "variable to enable AI responses."
            )

        try:
            # Use the Chat Completions API (sync)
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            # Navigate response structure safely
            choices = getattr(resp, "choices", None) or resp.get("choices", [])
            if not choices:
                return "AI returned an empty response."
            message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
            if isinstance(message, dict):
                return message.get("content", "")
            # Fallback for different clients
            return str(choices[0])
        except Exception as e:
            return f"AI service error: {e}"
