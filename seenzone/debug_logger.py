"""
Debug Logger Module
====================
Centralized logging utilities for SeenZone debugging.

Provides console.log-style output for:
- Timing measurements (CV, inference, cycle)
- Cue detection
- Rule evaluation
- State transitions
- Response generation
"""

import time
from typing import List, Optional, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

# Set to False to disable debug output
DEBUG_ENABLED = True

# Color codes for terminal output (ANSI)
COLORS = {
    "TIMING": "\033[36m",    # Cyan
    "CUE": "\033[33m",       # Yellow
    "RULE": "\033[35m",      # Magenta
    "STATE": "\033[32m",     # Green
    "RESPONSE": "\033[34m",  # Blue
    "SKIP": "\033[90m",      # Gray
    "WARN": "\033[91m",      # Red
    "RESET": "\033[0m",      # Reset
}


# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def log_timing(phase: str, duration_ms: float, extra: str = "") -> None:
    """
    Log timing information for a processing phase.
    
    Args:
        phase: Name of the phase (e.g., "CV", "Inference", "Cycle")
        duration_ms: Duration in milliseconds
        extra: Optional extra info (e.g., FPS)
    """
    if not DEBUG_ENABLED:
        return
    
    color = COLORS["TIMING"]
    reset = COLORS["RESET"]
    extra_str = f" {extra}" if extra else ""
    print(f"{color}[TIMING] {phase}: {duration_ms:.1f}ms{extra_str}{reset}")


def log_cues(cues: List[str]) -> None:
    """
    Log detected cues from CV processing.
    
    Args:
        cues: List of FOL predicates like ["Present(user)", "Engaged(user)"]
    """
    if not DEBUG_ENABLED:
        return
    
    color = COLORS["CUE"]
    reset = COLORS["RESET"]
    
    if cues:
        cue_str = ", ".join(cues)
        print(f"{color}[CUES] {cue_str}{reset}")
    else:
        print(f"{color}[CUES] (none detected){reset}")


def log_rule_eval(rule_name: str, matched: bool, explanation: str = "") -> None:
    """
    Log rule evaluation result.
    
    Args:
        rule_name: Name of the rule
        matched: Whether the rule matched
        explanation: Optional explanation
    """
    if not DEBUG_ENABLED:
        return
    
    color = COLORS["RULE"]
    reset = COLORS["RESET"]
    
    status = "MATCH" if matched else "FAIL"
    exp_str = f" ({explanation})" if explanation else ""
    print(f"{color}[RULE] {rule_name}: {status}{exp_str}{reset}")


def log_state_change(old_state: Any, new_state: Any, rule_name: Optional[str] = None) -> None:
    """
    Log a state transition.
    
    Args:
        old_state: Previous emotional state
        new_state: New emotional state
        rule_name: Name of the rule that caused the transition
    """
    if not DEBUG_ENABLED:
        return
    
    color = COLORS["STATE"]
    reset = COLORS["RESET"]
    
    old_name = old_state.name if hasattr(old_state, 'name') else str(old_state)
    new_name = new_state.name if hasattr(new_state, 'name') else str(new_state)
    rule_str = f" (rule: {rule_name})" if rule_name else ""
    
    print(f"{color}[STATE] {old_name} → {new_name}{rule_str}{reset}")


def log_response(response: str, source: str = "LLM") -> None:
    """
    Log a generated response.
    
    Args:
        response: The response text
        source: "LLM" or "fallback"
    """
    if not DEBUG_ENABLED:
        return
    
    color = COLORS["RESPONSE"]
    reset = COLORS["RESET"]
    
    # Truncate long responses
    display = response[:80] + "..." if len(response) > 80 else response
    print(f"{color}[RESPONSE] ({source}) {display}{reset}")


def log_skip(reason: str) -> None:
    """
    Log a skipped action.
    
    Args:
        reason: Why the action was skipped
    """
    if not DEBUG_ENABLED:
        return
    
    color = COLORS["SKIP"]
    reset = COLORS["RESET"]
    print(f"{color}[SKIP] {reason}{reset}")


def log_warn(message: str) -> None:
    """
    Log a warning.
    
    Args:
        message: Warning message
    """
    if not DEBUG_ENABLED:
        return
    
    color = COLORS["WARN"]
    reset = COLORS["RESET"]
    print(f"{color}[WARN] {message}{reset}")


def log_cycle_summary(cycle: int, fps: float, state: Any, cue_count: int) -> None:
    """
    Log a summary line for the current cycle.
    
    Args:
        cycle: Cycle number
        fps: Current frames per second
        state: Current emotional state
        cue_count: Number of active cues
    """
    if not DEBUG_ENABLED:
        return
    
    state_name = state.name if hasattr(state, 'name') else str(state)
    print(f"[Cycle {cycle}] FPS: {fps:.1f} | State: {state_name} | Cues: {cue_count}")


# =============================================================================
# TIMING CONTEXT MANAGER
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, auto_log: bool = True):
        self.name = name
        self.auto_log = auto_log
        self.start_time: float = 0
        self.duration_ms: float = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000
        if self.auto_log:
            log_timing(self.name, self.duration_ms)
