"""
Reasoning module for SeenZone agent.
Implements FOL-inspired forward-chaining inference.
"""

from .knowledge_base import KnowledgeBase, Predicate
from .rules import Rule, RuleEngine, create_default_rules
from .inference import InferenceEngine, InferenceResult

__all__ = [
    "KnowledgeBase",
    "Predicate", 
    "Rule",
    "RuleEngine",
    "create_default_rules",
    "InferenceEngine",
    "InferenceResult",
]
