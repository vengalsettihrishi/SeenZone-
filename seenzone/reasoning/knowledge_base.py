"""
Knowledge Base Module
======================
Implements a simple knowledge base for storing and querying facts.

The knowledge base stores ground predicates (facts) derived from
the CV module's symbolic output. It supports:
- Adding/removing facts
- Querying whether predicates hold (including negation)
- Clearing all facts between cycles

Academic Mapping:
- Knowledge Base (KR): Set of ground facts
- Closed World Assumption: Unstated facts are false
"""

from dataclasses import dataclass
from typing import Set, List, Optional


# =============================================================================
# PREDICATE REPRESENTATION
# =============================================================================

@dataclass(frozen=True)
class Predicate:
    """
    Represents a predicate in the knowledge representation system.
    
    Uses frozen dataclass for immutability and hashability.
    This allows predicates to be used in sets and as dict keys.
    
    Attributes:
        name: The predicate name (e.g., "Present", "Engaged")
        negated: Whether this is a negative literal (NOT predicate)
    
    Examples:
        Predicate("Present")           # Present is true
        Predicate("Engaged", negated=True)  # NOT Engaged
    """
    name: str
    negated: bool = False
    
    def __str__(self) -> str:
        """Human-readable representation."""
        if self.negated:
            return f"NOT {self.name}"
        return self.name
    
    def __repr__(self) -> str:
        return f"Predicate({self.name!r}, negated={self.negated})"
    
    def negate(self) -> "Predicate":
        """Return the negation of this predicate."""
        return Predicate(self.name, negated=not self.negated)
    
    @classmethod
    def from_string(cls, s: str) -> "Predicate":
        """
        Parse a predicate from string format.
        
        Supports formats:
        - "Present" → Predicate("Present")
        - "NOT Engaged" → Predicate("Engaged", negated=True)
        - "Present(user)" → Predicate("Present")  # Ignores args for simplicity
        """
        s = s.strip()
        negated = False
        
        if s.upper().startswith("NOT "):
            negated = True
            s = s[4:].strip()
        
        # Remove parentheses and arguments if present
        if "(" in s:
            s = s[:s.index("(")]
        
        return cls(name=s, negated=negated)


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class KnowledgeBase:
    """
    Simple knowledge base for storing ground facts.
    
    Implements the Closed World Assumption:
    - Facts in the KB are true
    - Facts not in the KB are assumed false
    
    The KB is updated each agent cycle with new predicates
    from the CV module.
    
    Academic Mapping:
    - Ground Atoms: Predicate names stored as strings
    - CWA: Query for absent fact returns False
    """
    
    def __init__(self):
        """Initialize an empty knowledge base."""
        self._facts: Set[str] = set()
        self._history: List[Set[str]] = []  # For debugging
        print("[KnowledgeBase] Initialized (empty)")
    
    @property
    def facts(self) -> Set[str]:
        """Get current set of facts (read-only view)."""
        return frozenset(self._facts)
    
    def add_fact(self, name: str) -> None:
        """
        Add a fact to the knowledge base.
        
        Args:
            name: Predicate name (e.g., "Present", "Engaged")
        """
        # Clean the name (remove parentheses if present)
        if "(" in name:
            name = name[:name.index("(")]
        self._facts.add(name)
    
    def remove_fact(self, name: str) -> None:
        """
        Remove a fact from the knowledge base.
        
        Args:
            name: Predicate name to remove
        """
        if "(" in name:
            name = name[:name.index("(")]
        self._facts.discard(name)
    
    def holds(self, predicate: Predicate) -> bool:
        """
        Check if a predicate holds in the current knowledge base.
        
        Implements the Closed World Assumption:
        - If predicate.negated is False: returns True iff fact is in KB
        - If predicate.negated is True: returns True iff fact is NOT in KB
        
        Args:
            predicate: The predicate to query
            
        Returns:
            True if the predicate holds, False otherwise
        """
        fact_present = predicate.name in self._facts
        
        if predicate.negated:
            return not fact_present
        else:
            return fact_present
    
    def query(self, name: str) -> bool:
        """
        Simple query by predicate name.
        
        Args:
            name: Predicate name to check
            
        Returns:
            True if fact is in KB
        """
        if "(" in name:
            name = name[:name.index("(")]
        return name in self._facts
    
    def clear(self) -> None:
        """Clear all facts from the knowledge base."""
        if self._facts:
            self._history.append(self._facts.copy())
        self._facts.clear()
    
    def update_from_predicates(self, fol_predicates: List[str]) -> None:
        """
        Update KB from a list of FOL predicate strings.
        
        Clears existing facts and adds new ones.
        
        Args:
            fol_predicates: List like ["Present(user)", "Engaged(user)"]
        """
        self.clear()
        for pred_str in fol_predicates:
            self.add_fact(pred_str)
    
    def get_state(self) -> dict:
        """Get current KB state for debugging."""
        return {
            "facts": sorted(list(self._facts)),
            "count": len(self._facts)
        }
    
    def __len__(self) -> int:
        return len(self._facts)
    
    def __contains__(self, name: str) -> bool:
        return self.query(name)
    
    def __repr__(self) -> str:
        return f"KnowledgeBase({sorted(self._facts)})"
