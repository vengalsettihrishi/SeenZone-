"""
Inference Engine Module
========================
Implements forward-chaining inference for state transitions.

The inference engine coordinates:
1. Knowledge Base: Current facts from CV
2. Rule Engine: Rule evaluation and priority resolution
3. State Space: Executing valid state transitions

It also maintains explanation traces for transparency.

Academic Mapping:
- Forward Chaining: Data-driven inference from facts to conclusions
- Inference Trace: Audit trail of reasoning steps
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from .knowledge_base import KnowledgeBase, Predicate
from .rules import RuleEngine, Rule, create_default_rules

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seenzone.state_space import StateSpace, EmotionalState


# =============================================================================
# INFERENCE RESULT
# =============================================================================

@dataclass
class InferenceResult:
    """
    Result of an inference cycle.
    
    Captures the outcome and reasoning trace for transparency.
    """
    fired_rule: Optional[Rule]
    new_state: Optional[EmotionalState]
    matching_rules: List[Rule]
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def transitioned(self) -> bool:
        """Whether a state transition occurred."""
        return self.new_state is not None


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Forward-chaining inference engine for state transitions.
    
    Responsibilities:
    1. Update knowledge base with new facts
    2. Evaluate rules to find matches
    3. Select highest-priority matching rule
    4. Request state transition (if valid)
    5. Record explanation trace
    6. THROTTLE transitions to prevent rapid oscillation
    
    Academic Mapping:
    - Inference: KB + Rules → Conclusion
    - Forward Chaining: Start with facts, derive new states
    - Explanation: Audit trail of rule firings
    """
    
    # Minimum time in state before allowing transition
    MIN_STATE_DURATION = 2.0  # seconds
    
    # Priority threshold to bypass throttle (emergency/override rules)
    PRIORITY_BYPASS_THRESHOLD = 7
    
    # Time without valence before positive state can decay
    VALENCE_DECAY_DURATION = 3.0  # seconds
    
    def __init__(self, state_space: Optional[StateSpace] = None, 
                 load_default_rules: bool = True):
        """
        Initialize the inference engine.
        
        Args:
            state_space: Optional state space for transition validation
            load_default_rules: Whether to load default rule set
        """
        self.kb = KnowledgeBase()
        self.rule_engine = RuleEngine()
        self.state_space = state_space
        
        # Inference history for explanation
        self._history: List[InferenceResult] = []
        self._last_result: Optional[InferenceResult] = None
        
        # State transition throttling
        self._state_entry_time: Optional[float] = None
        self._current_tracked_state: Optional[EmotionalState] = None
        
        # Valence decay tracking (for positive state maintenance)
        self._last_valence_time: Optional[float] = None
        
        # Load default rules if requested
        if load_default_rules:
            self.rule_engine.add_rules(create_default_rules())
        
        print(f"[InferenceEngine] Initialized with {len(self.rule_engine)} rules")
        print(f"[InferenceEngine] Throttle: {self.MIN_STATE_DURATION}s, valence decay: {self.VALENCE_DECAY_DURATION}s")
    
    def set_state_space(self, state_space: StateSpace) -> None:
        """Set the state space reference."""
        self.state_space = state_space
    
    def update_facts(self, fol_predicates: List[str]) -> None:
        """
        Update knowledge base with new predicates from CV.
        
        Clears previous facts and adds new ones.
        
        Args:
            fol_predicates: List like ["Present(user)", "Engaged(user)"]
        """
        self.kb.update_from_predicates(fol_predicates)
    
    def _get_time_in_state(self) -> float:
        """Get time spent in current state in seconds."""
        import time
        if self._state_entry_time is None:
            return 0.0
        return time.time() - self._state_entry_time
    
    def _update_state_tracking(self, new_state: EmotionalState) -> None:
        """Update state tracking when transitioning."""
        import time
        self._current_tracked_state = new_state
        self._state_entry_time = time.time()
    
    def infer(self) -> Optional[EmotionalState]:
        """
        Run inference and return new state if transition should occur.
        
        Steps:
        1. Evaluate all rules against current KB
        2. Select highest-priority matching rule
        3. CHECK THROTTLE: Block if in state < MIN_DURATION (unless high priority)
        4. Check if transition is valid in state space
        5. Return target state (or None if no transition)
        
        Returns:
            Target emotional state, or None if no transition
        """
        import time
        
        # Initialize state tracking if needed
        if self._state_entry_time is None and self.state_space:
            self._current_tracked_state = self.state_space.current_state
            self._state_entry_time = time.time()
        
        # Log current KB facts for visibility
        if self.kb.facts:
            print(f"[KB] Facts: {sorted(self.kb.facts)}")
        
        # Get all matching rules (sorted by priority)
        matching = self.rule_engine.evaluate(self.kb)
        
        # Log all rule evaluations for visibility
        for rule in self.rule_engine.rules:
            is_match = rule in matching
            if is_match:
                print(f"[RULE] {rule.name}: \033[32mMATCH\033[0m → {rule.target_state.name}")
            else:
                # Show which conditions failed
                failed = [str(c) for c in rule.conditions if not self.kb.holds(c)]
                if failed:
                    print(f"[RULE] {rule.name}: \033[90mFAIL\033[0m (missing: {', '.join(failed)})")
        
        # No matching rules
        if not matching:
            self._last_result = InferenceResult(
                fired_rule=None,
                new_state=None,
                matching_rules=[],
                explanation="No rules matched current knowledge base"
            )
            self._history.append(self._last_result)
            return None
        
        # Get highest priority rule
        best_rule = matching[0]
        target_state = best_rule.target_state
        
        # Check if transition is valid
        if self.state_space:
            current = self.state_space.current_state
            time_in_state = self._get_time_in_state()
            
            # === STATE HOLD: If staying in same state, log and return ===
            if target_state == current:
                # Log state hold for positive states
                if current == EmotionalState.S5_POSITIVE_STATE:
                    print(f"[STATE_HOLD] Maintaining {current.name} (engaged or expressive)")
                
                self._last_result = InferenceResult(
                    fired_rule=best_rule,
                    new_state=None,
                    matching_rules=matching,
                    explanation=f"STATE_HOLD: Staying in {current.name}"
                )
                self._history.append(self._last_result)
                return None
            
            # === THROTTLE CHECK: Block rapid transitions ===
            if time_in_state < self.MIN_STATE_DURATION:
                # Allow bypass for high-priority rules
                if best_rule.priority >= self.PRIORITY_BYPASS_THRESHOLD:
                    print(f"[STATE_THROTTLE] Bypassed (priority {best_rule.priority} >= {self.PRIORITY_BYPASS_THRESHOLD})")
                else:
                    print(f"[STATE_BLOCK] Transition blocked (duration={time_in_state:.1f}s < {self.MIN_STATE_DURATION}s)")
                    self._last_result = InferenceResult(
                        fired_rule=best_rule,
                        new_state=None,
                        matching_rules=matching,
                        explanation=f"Transition throttled: {time_in_state:.1f}s < {self.MIN_STATE_DURATION}s minimum"
                    )
                    self._history.append(self._last_result)
                    return None
            
            # Check if transition is valid in state space
            valid_targets = {t.to_state for t in self.state_space.get_valid_transitions()}
            if target_state not in valid_targets:
                self._last_result = InferenceResult(
                    fired_rule=best_rule,
                    new_state=None,
                    matching_rules=matching,
                    explanation=f"Transition to {target_state.name} not valid from {current.name}"
                )
                self._history.append(self._last_result)
                print(f"[INFERENCE] Transition blocked - {target_state.name} not reachable from {current.name}")
                return None
        
        # Inference successful - update state tracking
        self._update_state_tracking(target_state)
        
        self._last_result = InferenceResult(
            fired_rule=best_rule,
            new_state=target_state,
            matching_rules=matching,
            explanation=f"Rule '{best_rule.name}' fired: {' ∧ '.join(str(c) for c in best_rule.conditions)} → {target_state.name}"
        )
        self._history.append(self._last_result)
        
        print(f"[STATE] \033[32mTRANSITION\033[0m {self._current_tracked_state} → {target_state.name}")
        
        return target_state
    
    def get_explanation(self) -> str:
        """
        Get explanation of the last inference cycle.
        
        Returns:
            Human-readable explanation string
        """
        if not self._last_result:
            return "No inference has been run yet"
        
        result = self._last_result
        lines = [
            f"=== Inference Explanation ===",
            f"KB Facts: {sorted(self.kb.facts)}",
            f"Matching Rules: {len(result.matching_rules)}",
        ]
        
        if result.matching_rules:
            lines.append("  Matches (by priority):")
            for i, rule in enumerate(result.matching_rules[:3]):  # Top 3
                marker = "→" if rule == result.fired_rule else " "
                lines.append(f"    {marker} [{rule.priority}] {rule.name}")
        
        lines.append(f"Result: {result.explanation}")
        
        return "\n".join(lines)
    
    def get_verbose_explanation(self) -> str:
        """Get detailed explanation including all rule evaluations."""
        lines = [
            "=== Detailed Rule Evaluation ===",
            f"Current KB: {sorted(self.kb.facts)}",
            ""
        ]
        
        for explanation in self.rule_engine.get_all_explanations(self.kb):
            lines.append(f"  {explanation}")
        
        lines.append("")
        lines.append(self.get_explanation())
        
        return "\n".join(lines)
    
    def get_history(self, n: int = 5) -> List[InferenceResult]:
        """Get last N inference results."""
        return self._history[-n:]
    
    def clear_history(self) -> None:
        """Clear inference history."""
        self._history.clear()
        self._last_result = None
