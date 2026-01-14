"""
Groq LLM Client
================
Wrapper for the Groq API for LLM inference.

The LLM is strictly an ACTUATOR — it does not infer emotions or states.
All reasoning is done upstream by the KR + inference engine.

The client:
- Handles API authentication
- Manages request/response
- Provides graceful fallback on failure
"""

import os
from typing import Optional
from dataclasses import dataclass

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads from .env in project root
except ImportError:
    pass  # dotenv not installed, use system env vars

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[GroqClient] Warning: groq package not installed")


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = "llama-3.3-70b-versatile"  # High-quality model for emotional therapy
    max_tokens: int = 150          # Keep responses brief
    temperature: float = 0.7       # Some creativity
    timeout: float = 10.0          # Request timeout


class GroqClient:
    """
    Groq API client for LLM inference.
    
    Usage:
        client = GroqClient()
        if client.is_available():
            response = client.generate(prompt)
    
    The client reads GROQ_API_KEY from environment.
    If unavailable, is_available() returns False.
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[LLMConfig] = None):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client: Optional[Groq] = None
        self._available = False
        self._last_error: Optional[str] = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Groq client."""
        if not GROQ_AVAILABLE:
            self._last_error = "groq package not installed"
            print(f"[GroqClient] Not available: {self._last_error}")
            return
        
        if not self._api_key:
            self._last_error = "GROQ_API_KEY not set"
            print(f"[GroqClient] Not available: {self._last_error}")
            return
        
        try:
            self._client = Groq(api_key=self._api_key)
            self._available = True
            print(f"[GroqClient] Initialized (model={self.config.model})")
        except Exception as e:
            self._last_error = str(e)
            print(f"[GroqClient] Initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if the client is ready to generate."""
        return self._available and self._client is not None
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User/context prompt
            system_prompt: System prompt defining persona
            
        Returns:
            Generated text, or None if failed
        """
        if not self.is_available():
            return None
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            
            return None
            
        except Exception as e:
            self._last_error = str(e)
            print(f"[GroqClient] Generation failed: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test API connectivity with a simple prompt."""
        if not self.is_available():
            return False
        
        result = self.generate("Say 'OK' if you can hear me.", 
                               system_prompt="Respond with only 'OK'.")
        return result is not None
