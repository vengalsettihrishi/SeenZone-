"""
State-Space Model for Emotional States
=======================================
Implements the state-space search model for a goal-based agent.

State Space Definition:
- S0: Neutral           — Baseline state, no strong affect detected
- S1: Sadness_Detected  — Visual or textual sadness indicators
- S2: Depression_Suspected — Prolonged S1 with disengagement signals
- S3: High_Stress       — Stress indicators (posture, speech patterns)
- S4: Prolonged_Stress  — Sustained S3 state over time
- S5: Positive_State    — Positive affect indicators detected
- S6: Distressed_Silent — Visible distress but no verbal interaction
- S_goal: Emotionally_Stabilized — Target state for the agent

The agent's goal is to guide transitions toward S_goal through
appropriate empathetic responses conditioned on current state.

This module provides:
1. EmotionalState enum - all possible states
2. StateTransition - rules for valid transitions
3. StateSpace - manages current state and transition logic
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Set, Callable, Any
from datetime import datetime


# =============================================================================
# EMOTIONAL STATE DEFINITIONS
# =============================================================================

class EmotionalState(Enum):
    """
    Enumeration of all possible emotional states in the state space.
    
    These states represent *inferred affective contexts*, not clinical
    diagnoses. They are derived from symbolic cues extracted by the
    CV module and textual analysis.
    
    State Semantics:
    - S0 (Neutral): Default/baseline. No significant affective indicators.
    - S1 (Sadness): Detected sadness cues (facial, postural, textual).
    - S2 (Depression Suspected): Prolonged S1 + disengagement patterns.
    - S3 (High Stress): Acute stress indicators.
    - S4 (Prolonged Stress): Sustained stress over multiple cycles.
    - S5 (Positive): Positive affect indicators present.
    - S6 (Distressed Silent): Distress visible but user not communicating.
    - S_goal (Stabilized): Target state - user appears emotionally stable.
    """
    S0_NEUTRAL = auto()
    S1_SADNESS_DETECTED = auto()
    S2_DEPRESSION_SUSPECTED = auto()
    S3_HIGH_STRESS = auto()
    S4_PROLONGED_STRESS = auto()
    S5_POSITIVE_STATE = auto()
    S6_DISTRESSED_SILENT = auto()
    S_GOAL_STABILIZED = auto()
    
    def __str__(self) -> str:
        """Human-readable state name."""
        names = {
            self.S0_NEUTRAL: "Neutral",
            self.S1_SADNESS_DETECTED: "Sadness Detected",
            self.S2_DEPRESSION_SUSPECTED: "Depression Suspected",
            self.S3_HIGH_STRESS: "High Stress",
            self.S4_PROLONGED_STRESS: "Prolonged Stress",
            self.S5_POSITIVE_STATE: "Positive State",
            self.S6_DISTRESSED_SILENT: "Distressed & Silent",
            self.S_GOAL_STABILIZED: "Emotionally Stabilized",
        }
        return names.get(self, self.name)
    
    @property
    def is_goal_state(self) -> bool:
        """Check if this is the goal state."""
        return self == EmotionalState.S_GOAL_STABILIZED
    
    @property
    def severity_level(self) -> int:
        """
        Severity level for state comparison (higher = more concerning).
        Used to determine if transitions move toward or away from goal.
        """
        severity = {
            self.S_GOAL_STABILIZED: 0,
            self.S5_POSITIVE_STATE: 1,
            self.S0_NEUTRAL: 2,
            self.S1_SADNESS_DETECTED: 3,
            self.S3_HIGH_STRESS: 4,
            self.S6_DISTRESSED_SILENT: 5,
            self.S4_PROLONGED_STRESS: 6,
            self.S2_DEPRESSION_SUSPECTED: 7,
        }
        return severity.get(self, 5)


# =============================================================================
# STATE TRANSITION DEFINITION
# =============================================================================

@dataclass
class StateTransition:
    """
    Represents a valid state transition in the state space.
    
    Each transition has:
    - from_state: Origin state
    - to_state: Destination state
    - condition_name: Human-readable description of trigger condition
    - priority: Higher priority transitions are checked first
    
    The actual condition logic is evaluated by the reasoning module (Sprint 3).
    This structure just defines what transitions are valid.
    """
    from_state: EmotionalState
    to_state: EmotionalState
    condition_name: str
    priority: int = 0
    
    def is_toward_goal(self) -> bool:
        """Check if this transition moves toward the goal state."""
        return self.to_state.severity_level < self.from_state.severity_level


# =============================================================================
# STATE SPACE MANAGER
# =============================================================================

@dataclass
class StateSpace:
    """
    Manages the emotional state space and transition logic.
    
    This implements a finite state machine where:
    - States are from EmotionalState enum
    - Transitions are governed by rules (defined in Sprint 3)
    - The goal is to reach S_GOAL_STABILIZED
    
    The StateSpace tracks:
    - Current state
    - State history (for pattern detection)
    - Valid transitions from current state
    
    Academic Mapping:
    - State space: Set of all EmotionalState values
    - Initial state: S0_NEUTRAL
    - Goal state: S_GOAL_STABILIZED
    - Operators: State transitions triggered by cues
    """
    current_state: EmotionalState = EmotionalState.S0_NEUTRAL
    history: List[tuple] = field(default_factory=list)  # (state, timestamp)
    transitions: List[StateTransition] = field(default_factory=list)
    max_history: int = 50
    
    def __post_init__(self):
        """Initialize with default transition definitions."""
        self._define_default_transitions()
        self._record_state()
    
    def _define_default_transitions(self) -> None:
        """
        Define all valid state transitions.
        
        These transitions form the edges of our state-space graph.
        Actual triggering conditions are evaluated by the reasoning module.
        """
        transitions = [
            # From Neutral
            StateTransition(EmotionalState.S0_NEUTRAL, EmotionalState.S1_SADNESS_DETECTED, 
                          "sadness_cues_detected", priority=1),
            StateTransition(EmotionalState.S0_NEUTRAL, EmotionalState.S3_HIGH_STRESS, 
                          "stress_cues_detected", priority=1),
            StateTransition(EmotionalState.S0_NEUTRAL, EmotionalState.S5_POSITIVE_STATE, 
                          "positive_cues_detected", priority=1),
            StateTransition(EmotionalState.S0_NEUTRAL, EmotionalState.S_GOAL_STABILIZED, 
                          "stability_confirmed", priority=2),
            
            # From Sadness Detected
            StateTransition(EmotionalState.S1_SADNESS_DETECTED, EmotionalState.S0_NEUTRAL, 
                          "sadness_resolved", priority=1),
            StateTransition(EmotionalState.S1_SADNESS_DETECTED, EmotionalState.S2_DEPRESSION_SUSPECTED, 
                          "prolonged_sadness_disengagement", priority=2),
            StateTransition(EmotionalState.S1_SADNESS_DETECTED, EmotionalState.S6_DISTRESSED_SILENT, 
                          "distress_no_response", priority=2),
            StateTransition(EmotionalState.S1_SADNESS_DETECTED, EmotionalState.S5_POSITIVE_STATE, 
                          "mood_improvement", priority=1),
            
            # From Depression Suspected
            StateTransition(EmotionalState.S2_DEPRESSION_SUSPECTED, EmotionalState.S1_SADNESS_DETECTED, 
                          "engagement_resumed", priority=1),
            StateTransition(EmotionalState.S2_DEPRESSION_SUSPECTED, EmotionalState.S0_NEUTRAL, 
                          "significant_improvement", priority=2),
            
            # From High Stress
            StateTransition(EmotionalState.S3_HIGH_STRESS, EmotionalState.S0_NEUTRAL, 
                          "stress_resolved", priority=1),
            StateTransition(EmotionalState.S3_HIGH_STRESS, EmotionalState.S4_PROLONGED_STRESS, 
                          "stress_sustained", priority=2),
            StateTransition(EmotionalState.S3_HIGH_STRESS, EmotionalState.S1_SADNESS_DETECTED, 
                          "stress_to_sadness", priority=1),
            
            # From Prolonged Stress
            StateTransition(EmotionalState.S4_PROLONGED_STRESS, EmotionalState.S3_HIGH_STRESS, 
                          "stress_reduced", priority=1),
            StateTransition(EmotionalState.S4_PROLONGED_STRESS, EmotionalState.S0_NEUTRAL, 
                          "stress_fully_resolved", priority=2),
            
            # From Positive State
            StateTransition(EmotionalState.S5_POSITIVE_STATE, EmotionalState.S_GOAL_STABILIZED, 
                          "sustained_positive", priority=2),
            StateTransition(EmotionalState.S5_POSITIVE_STATE, EmotionalState.S0_NEUTRAL, 
                          "positive_faded", priority=1),
            
            # From Distressed Silent
            StateTransition(EmotionalState.S6_DISTRESSED_SILENT, EmotionalState.S1_SADNESS_DETECTED, 
                          "engagement_started", priority=1),
            StateTransition(EmotionalState.S6_DISTRESSED_SILENT, EmotionalState.S2_DEPRESSION_SUSPECTED, 
                          "prolonged_silence", priority=2),
            
            # From Goal State (can regress)
            StateTransition(EmotionalState.S_GOAL_STABILIZED, EmotionalState.S0_NEUTRAL, 
                          "minor_fluctuation", priority=1),
            StateTransition(EmotionalState.S_GOAL_STABILIZED, EmotionalState.S5_POSITIVE_STATE, 
                          "continued_positive", priority=1),
        ]
        self.transitions = transitions
    
    def _record_state(self) -> None:
        """Record current state in history."""
        self.history.append((self.current_state, datetime.now()))
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_valid_transitions(self) -> List[StateTransition]:
        """Get all valid transitions from current state."""
        return [t for t in self.transitions if t.from_state == self.current_state]
    
    def transition_to(self, new_state: EmotionalState) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition was valid and executed, False otherwise
        """
        # Check if transition is valid
        valid_targets = {t.to_state for t in self.get_valid_transitions()}
        
        if new_state not in valid_targets:
            print(f"[StateSpace] Invalid transition: {self.current_state} -> {new_state}")
            return False
        
        old_state = self.current_state
        self.current_state = new_state
        self._record_state()
        
        print(f"[StateSpace] Transition: {old_state} -> {new_state}")
        return True
    
    def get_state_duration(self) -> float:
        """Get how long (seconds) we've been in the current state."""
        if len(self.history) < 2:
            return 0.0
        
        # Find when we entered current state
        for i in range(len(self.history) - 2, -1, -1):
            if self.history[i][0] != self.current_state:
                entry_time = self.history[i + 1][1]
                return (datetime.now() - entry_time).total_seconds()
        
        # Been in this state since beginning
        return (datetime.now() - self.history[0][1]).total_seconds()
    
    def is_at_goal(self) -> bool:
        """Check if currently at goal state."""
        return self.current_state.is_goal_state
    
    def distance_to_goal(self) -> int:
        """
        Heuristic distance to goal state based on severity.
        Used for state-space search evaluation.
        """
        return self.current_state.severity_level
    
    def get_state_summary(self) -> dict:
        """Get a summary of current state space status."""
        return {
            "current_state": str(self.current_state),
            "is_at_goal": self.is_at_goal(),
            "distance_to_goal": self.distance_to_goal(),
            "state_duration_seconds": self.get_state_duration(),
            "history_length": len(self.history),
            "valid_transitions": [str(t.to_state) for t in self.get_valid_transitions()]
        }
