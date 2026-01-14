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
    
    Academic Mapping:
    - Inference: KB + Rules → Conclusion
    - Forward Chaining: Start with facts, derive new states
    - Explanation: Audit trail of rule firings
    """
    
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
        
        # Load default rules if requested
        if load_default_rules:
            self.rule_engine.add_rules(create_default_rules())
        
        print(f"[InferenceEngine] Initialized with {len(self.rule_engine)} rules")
    
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
    
    def infer(self) -> Optional[EmotionalState]:
        """
        Run inference and return new state if transition should occur.
        
        Steps:
        1. Evaluate all rules against current KB
        2. Select highest-priority matching rule
        3. Check if transition is valid in state space
        4. Return target state (or None if no transition)
        
        Returns:
            Target emotional state, or None if no transition
        """
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
            
            # Don't transition to same state
            if target_state == current:
                self._last_result = InferenceResult(
                    fired_rule=best_rule,
                    new_state=None,
                    matching_rules=matching,
                    explanation=f"Rule '{best_rule.name}' matched but already in {current.name}"
                )
                self._history.append(self._last_result)
                print(f"[INFERENCE] No transition - already in {current.name}")
                return None
            
            # Check if transition is valid in state space
            valid_targets = {t.to_state for t in self.state_space.get_valid_transitions()}
            if target_state not in valid_targets:
                self._last_result = InferenceResult(
                    fired_rule=best_rule,
                    new_state=None,
                    matching_rules=matching,
                    explanation=f"Rule '{best_rule.name}' matched but transition to {target_state.name} not valid from {current.name}"
                )
                self._history.append(self._last_result)
                print(f"[INFERENCE] Transition blocked - {target_state.name} not reachable from {current.name}")
                return None
        
        # Inference successful
        self._last_result = InferenceResult(
            fired_rule=best_rule,
            new_state=target_state,
            matching_rules=matching,
            explanation=f"Rule '{best_rule.name}' fired: {' ∧ '.join(str(c) for c in best_rule.conditions)} → {target_state.name}"
        )
        self._history.append(self._last_result)
        
        print(f"[INFERENCE] \033[32mFIRING\033[0m {best_rule.name} → {target_state.name}")
        
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
