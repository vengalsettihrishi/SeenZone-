"""
Prompt Builder Module
======================
Constructs LLM prompts from emotional state and symbolic cues.

Key Design Principles:
1. State is AUTHORITATIVE — it comes from the reasoning engine
2. Cues are SUPPORTING — they provide context only
3. Response style is STATE-DEPENDENT — each state has a policy

The LLM never infers emotions — it only generates language
based on already-determined state.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seenzone.state_space import EmotionalState


# =============================================================================
# RESPONSE POLICIES
# =============================================================================

@dataclass
class ResponsePolicy:
    """
    Defines how the LLM should respond for a given state.
    
    Each emotional state has an associated policy that guides
    the LLM's tone, approach, and constraints.
    """
    style: str           # Brief description of response style
    tone: str            # Emotional tone to use
    approach: str        # What kind of response to generate
    avoid: str = ""      # What to avoid


# State → Response Policy mapping
STATE_POLICIES: Dict[EmotionalState, ResponsePolicy] = {
    EmotionalState.S0_NEUTRAL: ResponsePolicy(
        style="casual presence",
        tone="warm, relaxed",
        approach="Light acknowledgment of presence",
        avoid="Probing questions"
    ),
    EmotionalState.S1_SADNESS_DETECTED: ResponsePolicy(
        style="gentle validation",
        tone="soft, caring",
        approach="Acknowledge without assuming, offer presence",
        avoid="Forcing conversation, toxic positivity"
    ),
    EmotionalState.S2_DEPRESSION_SUSPECTED: ResponsePolicy(
        style="patient support",
        tone="calm, non-judgmental",
        approach="Express availability without pressure",
        avoid="Advice-giving, probing deeply"
    ),
    EmotionalState.S3_HIGH_STRESS: ResponsePolicy(
        style="grounding and calming",
        tone="steady, reassuring",
        approach="Acknowledge stress, offer grounding",
        avoid="Adding urgency or pressure"
    ),
    EmotionalState.S4_PROLONGED_STRESS: ResponsePolicy(
        style="sustained support",
        tone="patient, understanding",
        approach="Recognize ongoing difficulty, suggest small steps",
        avoid="Overwhelming suggestions"
    ),
    EmotionalState.S5_POSITIVE_STATE: ResponsePolicy(
        style="encouraging warmth",
        tone="bright, affirming",
        approach="Match positive energy, light celebration",
        avoid="Forced enthusiasm"
    ),
    EmotionalState.S6_DISTRESSED_SILENT: ResponsePolicy(
        style="gentle re-engagement",
        tone="soft, non-intrusive",
        approach="Express presence without demanding response",
        avoid="Pressure to talk, direct questions"
    ),
    EmotionalState.S_GOAL_STABILIZED: ResponsePolicy(
        style="light affirmation",
        tone="warm, content",
        approach="Acknowledge stability, simple presence",
        avoid="Over-analysis"
    ),
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are SeenZone, a supportive conversational companion.

Your role:
- Provide warm, empathetic presence
- Respond to the user's emotional state (provided to you)
- Be brief (1-2 sentences maximum)
- Be genuine, not performative

Strict constraints:
- NEVER diagnose emotions, mental health, or medical conditions
- NEVER claim certainty about how the user feels
- NEVER say "I can see you're sad/stressed/happy"
- Instead, use phrases like "It seems like..." or "I'm here if..."
- NEVER give unsolicited advice or solutions
- NEVER be preachy or lecture

The emotional state provided to you is authoritative — it comes from 
the system's reasoning engine. Your job is to generate an appropriate
response based on that state. Observations are supporting context only.

Respond naturally, like a caring friend checking in."""


# =============================================================================
# PROMPT BUILDER
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in conversation history."""
    role: str  # "assistant" or "user"
    content: str


class PromptBuilder:
    """
    Builds prompts for the LLM from state and cues.
    
    Prompt Structure:
    1. System prompt (persona + constraints)
    2. Authoritative state context
    3. Supporting observations
    4. Response policy guidance
    5. Conversation history (last 1-2 turns)
    """
    
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        """Initialize with system prompt."""
        self.system_prompt = system_prompt
        self._history: List[ConversationTurn] = []
        self._max_history = 2  # Last 2 turns only
        
        print("[PromptBuilder] Initialized")
    
    def add_assistant_turn(self, content: str) -> None:
        """Record an assistant response in history."""
        self._history.append(ConversationTurn("assistant", content))
        self._trim_history()
    
    def add_user_turn(self, content: str) -> None:
        """Record a user message in history."""
        self._history.append(ConversationTurn("user", content))
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Keep only the last N turns."""
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
    
    def get_state_context(self, state: EmotionalState) -> str:
        """
        Generate the authoritative state context section.
        
        This is the PRIMARY context — the LLM must respect this.
        """
        state_name = str(state)
        policy = STATE_POLICIES.get(state, STATE_POLICIES[EmotionalState.S0_NEUTRAL])
        
        context = f"""PRIMARY CONTEXT (authoritative):
- Emotional State: {state_name}
- This state was inferred by the reasoning system and is authoritative.

Response Policy for {state_name}:
- Style: {policy.style}
- Tone: {policy.tone}
- Approach: {policy.approach}"""
        
        if policy.avoid:
            context += f"\n- Avoid: {policy.avoid}"
        
        return context
    
    def get_observations_context(self, cues: List[str]) -> str:
        """
        Generate the supporting observations section.
        
        These are SECONDARY — they provide context but don't override state.
        """
        if not cues:
            return "SUPPORTING OBSERVATIONS: None available"
        
        # Clean cue names (remove parentheses)
        clean_cues = []
        for cue in cues:
            if "(" in cue:
                cue = cue[:cue.index("(")]
            clean_cues.append(cue.lower().replace("_", " "))
        
        return f"""SUPPORTING OBSERVATIONS (context only):
- {', '.join(clean_cues)}
- These observations support the state above but do not override it."""
    
    def get_history_context(self) -> str:
        """Generate conversation history section."""
        if not self._history:
            return "CONVERSATION HISTORY: None (new conversation)"
        
        lines = ["RECENT CONVERSATION:"]
        for turn in self._history:
            role = "You" if turn.role == "assistant" else "User"
            lines.append(f"- {role}: {turn.content}")
        
        return "\n".join(lines)
    
    def build(self, 
              state: EmotionalState, 
              cues: List[str],
              include_history: bool = True) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            state: Current emotional state (authoritative)
            cues: List of FOL predicates from CV
            include_history: Whether to include conversation history
            
        Returns:
            Complete prompt string
        """
        sections = [
            self.get_state_context(state),
            "",
            self.get_observations_context(cues),
        ]
        
        if include_history and self._history:
            sections.extend(["", self.get_history_context()])
        
        sections.extend([
            "",
            "Generate a brief (1-2 sentence) response appropriate for this state."
        ])
        
        return "\n".join(sections)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt
