"""
LLM Responder Actuator
=======================
Main actuator that generates empathetic responses via LLM.

This module implements the ACTUATOR component of PEAS.
The responder:
1. Receives the current emotional state (authoritative)
2. Receives supporting cues (context)
3. Builds a prompt using PromptBuilder
4. Generates response via GroqClient
5. Falls back to placeholder if LLM unavailable

Key Principle: The LLM is a LANGUAGE GENERATOR, not a REASONER.
All emotional inference has already happened upstream.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .llm_client import GroqClient, LLMConfig
from .prompt_builder import PromptBuilder, STATE_POLICIES

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seenzone.state_space import EmotionalState
from seenzone.peas import Actuator, ActuatorType


# =============================================================================
# PLACEHOLDER RESPONSES (Fallback)
# =============================================================================

# State-aware placeholder responses for when LLM is unavailable
PLACEHOLDER_RESPONSES: Dict[EmotionalState, str] = {
    EmotionalState.S0_NEUTRAL: 
        "I'm here with you.",
    EmotionalState.S1_SADNESS_DETECTED: 
        "I'm here if you'd like to talk. No pressure.",
    EmotionalState.S2_DEPRESSION_SUSPECTED: 
        "I'm with you. Take all the time you need.",
    EmotionalState.S3_HIGH_STRESS: 
        "Let's take a moment. I'm here.",
    EmotionalState.S4_PROLONGED_STRESS: 
        "I can see it's been a lot. One step at a time.",
    EmotionalState.S5_POSITIVE_STATE: 
        "It's good to see you. I'm glad you're here.",
    EmotionalState.S6_DISTRESSED_SILENT: 
        "I'm here whenever you're ready. No rush.",
    EmotionalState.S_GOAL_STABILIZED: 
        "You seem settled. I'm here if you need me.",
}


# =============================================================================
# LLM RESPONDER
# =============================================================================

class LLMResponder(Actuator):
    """
    LLM-based response generator implementing the Actuator interface.
    
    PEAS Mapping: ACTUATOR
    - Input: Action specification (state + cues)
    - Output: Natural language response
    
    The responder maintains:
    - GroqClient for LLM access
    - PromptBuilder for prompt construction
    - Fallback logic for graceful degradation
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM responder.
        
        Args:
            api_key: Groq API key (uses env var if not provided)
            config: LLM configuration
        """
        self.client = GroqClient(api_key=api_key, config=config)
        self.builder = PromptBuilder()
        
        self._use_llm = self.client.is_available()
        self._last_response: Optional[str] = None
        self._generation_count = 0
        self._fallback_count = 0
        
        status = "LLM enabled" if self._use_llm else "fallback mode"
        print(f"[LLMResponder] Initialized ({status})")
    
    @property
    def actuator_type(self) -> ActuatorType:
        """Returns the actuator type."""
        return ActuatorType.TEXT_RESPONSE
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available."""
        return self._use_llm and self.client.is_available()
    
    def execute(self, action: Dict[str, Any]) -> str:
        """
        Execute the actuator action (generate response).
        
        Required action keys:
        - state: EmotionalState
        - cues: List[str] (FOL predicates)
        
        Returns:
            Generated response text
        """
        state = action.get("state", EmotionalState.S0_NEUTRAL)
        cues = action.get("cues", [])
        
        return self.generate_response(state, cues)
    
    def generate_response(self, state: EmotionalState, cues: List[str]) -> str:
        """
        Generate an empathetic response for the given state.
        
        Args:
            state: Current emotional state (authoritative)
            cues: Supporting FOL predicates from CV
            
        Returns:
            Generated response text
        """
        # Try LLM generation
        if self.is_llm_available():
            response = self._generate_llm_response(state, cues)
            if response:
                self._generation_count += 1
                self._last_response = response
                
                # Record in history for context
                self.builder.add_assistant_turn(response)
                
                return response
        
        # Fallback to placeholder
        return self._get_fallback_response(state)
    
    def _generate_llm_response(self, state: EmotionalState, cues: List[str]) -> Optional[str]:
        """Generate response via LLM."""
        # Build prompt
        prompt = self.builder.build(state, cues)
        system_prompt = self.builder.get_system_prompt()
        
        # Generate
        response = self.client.generate(prompt, system_prompt)
        
        return response
    
    def _get_fallback_response(self, state: EmotionalState) -> str:
        """Get state-aware fallback response."""
        self._fallback_count += 1
        
        response = PLACEHOLDER_RESPONSES.get(
            state, 
            PLACEHOLDER_RESPONSES[EmotionalState.S0_NEUTRAL]
        )
        
        self._last_response = response
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get actuator statistics."""
        return {
            "llm_available": self.is_llm_available(),
            "llm_generations": self._generation_count,
            "fallback_responses": self._fallback_count,
            "last_response": self._last_response,
        }
    
    def add_user_message(self, message: str) -> None:
        """Record a user message in conversation history."""
        self.builder.add_user_turn(message)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.builder.clear_history()
