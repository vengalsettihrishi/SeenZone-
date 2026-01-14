"""
Rule Engine Module
==================
Implements forward-chaining rule evaluation for state transitions.

Rules are IF-THEN structures:
- IF: List of predicates that must hold
- THEN: Target emotional state to transition to

The rule engine:
1. Collects all rules whose conditions are satisfied
2. Sorts matching rules by priority
3. Returns the sorted list for the inference engine

Academic Mapping:
- Production Rules: IF conditions THEN action
- Forward Chaining: Data-driven inference
- Conflict Resolution: Priority-based selection
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .knowledge_base import KnowledgeBase, Predicate

# Import state definitions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seenzone.state_space import EmotionalState


# =============================================================================
# RULE DEFINITION
# =============================================================================

@dataclass
class Rule:
    """
    A production rule for state transitions.
    
    Structure: IF [conditions] THEN [target_state]
    
    Attributes:
        name: Human-readable rule identifier
        conditions: List of predicates that must hold
        target_state: Emotional state to transition to if rule fires
        priority: Higher priority rules are preferred (conflict resolution)
        description: Optional explanation of the rule's purpose
    
    Example:
        Rule(
            name="detect_sadness",
            conditions=[Predicate("HeadDown"), Predicate("Speaking", negated=True)],
            target_state=EmotionalState.S1_SADNESS_DETECTED,
            priority=2
        )
    """
    name: str
    conditions: List[Predicate]
    target_state: EmotionalState
    priority: int = 1
    description: str = ""
    
    def __str__(self) -> str:
        conds = " ∧ ".join(str(c) for c in self.conditions)
        return f"[{self.name}] IF ({conds}) THEN {self.target_state.name}"
    
    def __repr__(self) -> str:
        return f"Rule({self.name!r}, priority={self.priority})"
    
    def evaluate(self, kb: KnowledgeBase) -> bool:
        """
        Check if this rule's conditions are satisfied.
        
        All conditions must hold for the rule to match.
        
        Args:
            kb: Knowledge base to evaluate against
            
        Returns:
            True if all conditions hold, False otherwise
        """
        return all(kb.holds(cond) for cond in self.conditions)
    
    def get_explanation(self, kb: KnowledgeBase) -> str:
        """
        Generate explanation of why rule matched or didn't match.
        
        Returns:
            Human-readable explanation string
        """
        matched = []
        unmatched = []
        
        for cond in self.conditions:
            if kb.holds(cond):
                matched.append(f"✓ {cond}")
            else:
                unmatched.append(f"✗ {cond}")
        
        if unmatched:
            return f"Rule '{self.name}' NOT fired: {', '.join(unmatched)}"
        else:
            return f"Rule '{self.name}' FIRED: {', '.join(matched)} → {self.target_state.name}"


# =============================================================================
# RULE ENGINE
# =============================================================================

class RuleEngine:
    """
    Forward-chaining rule evaluation engine.
    
    Responsibilities:
    1. Store rule definitions
    2. Evaluate rules against knowledge base
    3. Return matching rules sorted by priority
    
    The engine does NOT execute transitions — that's the
    InferenceEngine's responsibility.
    
    Academic Mapping:
    - Conflict Set: All matching rules
    - Conflict Resolution: Priority ordering
    """
    
    def __init__(self):
        """Initialize with empty rule set."""
        self._rules: List[Rule] = []
        print("[RuleEngine] Initialized")
    
    @property
    def rules(self) -> List[Rule]:
        """Get list of all rules."""
        return self._rules.copy()
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self._rules.append(rule)
        print(f"[RuleEngine] Added rule: {rule.name} (priority={rule.priority})")
    
    def add_rules(self, rules: List[Rule]) -> None:
        """Add multiple rules."""
        for rule in rules:
            self.add_rule(rule)
    
    def evaluate(self, kb: KnowledgeBase) -> List[Rule]:
        """
        Evaluate all rules against the knowledge base.
        
        Returns all matching rules sorted by priority (highest first).
        
        Args:
            kb: Current knowledge base
            
        Returns:
            List of matching rules, sorted by priority descending
        """
        matching = [rule for rule in self._rules if rule.evaluate(kb)]
        
        # Sort by priority (highest first)
        matching.sort(key=lambda r: r.priority, reverse=True)
        
        return matching
    
    def get_best_match(self, kb: KnowledgeBase) -> Optional[Rule]:
        """
        Get the highest-priority matching rule.
        
        Args:
            kb: Current knowledge base
            
        Returns:
            Best matching rule, or None if no rules match
        """
        matches = self.evaluate(kb)
        return matches[0] if matches else None
    
    def get_all_explanations(self, kb: KnowledgeBase) -> List[str]:
        """Get explanations for all rules."""
        return [rule.get_explanation(kb) for rule in self._rules]
    
    def clear_rules(self) -> None:
        """Remove all rules."""
        self._rules.clear()
    
    def __len__(self) -> int:
        return len(self._rules)


# =============================================================================
# DEFAULT RULES
# =============================================================================

def create_default_rules() -> List[Rule]:
    """
    Create the default rule set for SeenZone.
    
    These rules map CV-derived predicates to emotional state transitions.
    Rules are ordered by priority for conflict resolution.
    
    NEW: Uses relative predicates (laptop-aware) and multi-signal detection.
    Positive override rules have higher priority than negative detection.
    
    Returns:
        List of default rules
    """
    rules = [
        # =================================================================
        # HIGHEST PRIORITY: POSITIVE OVERRIDE RULES (Priority 6-7)
        # These take precedence over negative state detection
        # =================================================================
        
        Rule(
            name="positive_override_expressive",
            conditions=[
                Predicate("Present"),
                Predicate("Smiling"),
                Predicate("ExpressiveFace"),
                Predicate("Active"),
            ],
            target_state=EmotionalState.S5_POSITIVE_STATE,
            priority=7,
            description="OVERRIDE: Smiling + expressive + active → positive (blocks sadness)"
        ),
        
        Rule(
            name="positive_smiling_engaged",
            conditions=[
                Predicate("Present"),
                Predicate("Smiling"),
                Predicate("Engaged"),
                Predicate("ShowsTension", negated=True),
            ],
            target_state=EmotionalState.S5_POSITIVE_STATE,
            priority=6,
            description="OVERRIDE: Smiling + engaged → positive"
        ),
        
        # =================================================================
        # HIGH PRIORITY: DISTRESS SIGNALS (Priority 5)
        # =================================================================
        
        Rule(
            name="detect_distressed_silent",
            conditions=[
                Predicate("Present"),
                Predicate("Engaged", negated=True),
                Predicate("LookingAway"),
            ],
            target_state=EmotionalState.S6_DISTRESSED_SILENT,
            priority=5,
            description="User present but disengaged and looking away"
        ),
        
        Rule(
            name="detect_high_stress",
            conditions=[
                Predicate("Present"),
                Predicate("ShowsTension"),
                Predicate("Attentive", negated=True),
            ],
            target_state=EmotionalState.S3_HIGH_STRESS,
            priority=4,
            description="User showing tension without attention"
        ),
        
        # =================================================================
        # MEDIUM PRIORITY: MOOD INDICATORS (Priority 3)
        # NEW: Multi-signal sadness requires BOTH relative head lowered AND low energy
        # =================================================================
        
        Rule(
            name="detect_sadness_robust",
            conditions=[
                Predicate("Present"),
                Predicate("HeadLoweredSignificantly"),  # RELATIVE to baseline
                Predicate("LowMotionEnergy"),            # Multi-signal requirement
                Predicate("Speaking", negated=True),
                Predicate("Smiling", negated=True),      # Block if smiling
            ],
            target_state=EmotionalState.S1_SADNESS_DETECTED,
            priority=3,
            description="ROBUST: Head lowered (relative) + low energy + not smiling → sadness"
        ),
        
        Rule(
            name="detect_sadness_disengaged",
            conditions=[
                Predicate("Present"),
                Predicate("HeadLoweredSignificantly"),
                Predicate("Engaged", negated=True),
                Predicate("ExpressiveFace", negated=True),
                Predicate("Smiling", negated=True),
            ],
            target_state=EmotionalState.S1_SADNESS_DETECTED,
            priority=3,
            description="Head lowered (relative) + disengaged + not expressive → sadness"
        ),
        
        Rule(
            name="detect_fatigue",
            conditions=[
                Predicate("Present"),
                Predicate("Fatigued"),
                Predicate("Attentive", negated=True),
            ],
            target_state=EmotionalState.S3_HIGH_STRESS,
            priority=3,
            description="User showing fatigue without attentiveness"
        ),
        
        # =================================================================
        # POSITIVE STATES (Priority 2)
        # =================================================================
        
        Rule(
            name="detect_positive",
            conditions=[
                Predicate("Present"),
                Predicate("Engaged"),
                Predicate("Attentive"),
                Predicate("ShowsTension", negated=True),
            ],
            target_state=EmotionalState.S5_POSITIVE_STATE,
            priority=2,
            description="User engaged and attentive without tension"
        ),
        
        Rule(
            name="confirm_stability",
            conditions=[
                Predicate("Present"),
                Predicate("Engaged"),
                Predicate("Attentive"),
                Predicate("ShowsTension", negated=True),
                Predicate("Fatigued", negated=True),
            ],
            target_state=EmotionalState.S_GOAL_STABILIZED,
            priority=1,
            description="User fully engaged, attentive, relaxed — goal state"
        ),
        
        # =================================================================
        # BASELINE (Priority 0)
        # NEW: Uses relative predicate instead of absolute HeadDown
        # =================================================================
        
        Rule(
            name="neutral_baseline",
            conditions=[
                Predicate("Present"),
                Predicate("ShowsTension", negated=True),
                Predicate("HeadLoweredSignificantly", negated=True),  # RELATIVE - laptop-aware
            ],
            target_state=EmotionalState.S0_NEUTRAL,
            priority=0,
            description="User present with neutral indicators (laptop-aware)"
        ),
    ]
    
    return rules
