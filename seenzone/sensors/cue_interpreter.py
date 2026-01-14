"""
Cue Interpreter - Symbolic Predicate Generation
=================================================
Converts raw affective cues from CVProcessor into symbolic predicates
that can be used by the FOL rule engine (Sprint 3).

This module bridges the gap between:
- Raw measurements (head_yaw = 0.35, eye_ar = 0.18)
- Symbolic predicates (looking_away = True, fatigued = True)

These predicates form the input to the knowledge representation system.
"""

from dataclasses import dataclass
from typing import Dict
from .cv_processor import AffectiveCues


# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================

@dataclass
class CueThresholds:
    """
    Thresholds for converting raw measurements to symbolic predicates.
    
    These values are tuned for typical webcam conditions.
    Adjust based on testing with actual users.
    """
    # Head pose thresholds
    yaw_looking_away: float = 0.25      # Absolute yaw > this = looking away
    pitch_head_down: float = -0.15      # Pitch < this = head down
    pitch_head_up: float = 0.15         # Pitch > this = head up
    
    # Eye thresholds
    eye_ar_closed: float = 0.15         # EAR < this = eyes closed
    eye_ar_fatigued: float = 0.20       # EAR < this = fatigued/drowsy
    eye_ar_wide: float = 0.35           # EAR > this = eyes wide open
    
    # Gaze thresholds
    gaze_centered_min: float = 0.35     # Gaze ratio between these = centered
    gaze_centered_max: float = 0.65
    
    # Mouth thresholds
    mouth_open_threshold: float = 0.3   # MAR > this = mouth open (speaking)
    mouth_tense_threshold: float = 0.05 # MAR < this = tense/clenched
    
    # Brow thresholds
    brow_raised_threshold: float = 0.6  # Brow height > this = raised
    brow_furrowed_threshold: float = 0.3  # Brow height < this = furrowed


# =============================================================================
# SYMBOLIC PREDICATES
# =============================================================================

@dataclass
class SymbolicPredicates:
    """
    Boolean predicates derived from affective cues.
    
    These serve as inputs to FOL rules for state transitions.
    Each predicate represents an observable behavioral signal.
    
    Naming Convention:
    - Use descriptive names that map to FOL predicates
    - e.g., is_engaged → Engaged(user)
    """
    # Engagement signals
    is_present: bool = False            # Face detected
    is_engaged: bool = False            # Looking at camera, attentive
    is_looking_away: bool = False       # Head turned away
    
    # Attention signals
    is_attentive: bool = False          # Engaged + eyes open
    is_distracted: bool = False         # Looking away or gaze off
    
    # Fatigue signals
    is_fatigued: bool = False           # Drowsy eyes
    eyes_closed: bool = False           # Eyes shut (blink or asleep)
    
    # Stress/affect signals
    shows_tension: bool = False         # Furrowed brow, clenched mouth
    brow_raised: bool = False           # Surprise/concern indicator
    
    # Head position
    head_down: bool = False             # Looking down (sadness indicator)
    head_up: bool = False               # Looking up
    
    # Verbal signals
    is_speaking: bool = False           # Mouth moving/open
    
    # Overall engagement level (composite)
    engagement_level: str = "unknown"   # "high", "medium", "low", "none"
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary for rule engine input."""
        return {
            "is_present": self.is_present,
            "is_engaged": self.is_engaged,
            "is_looking_away": self.is_looking_away,
            "is_attentive": self.is_attentive,
            "is_distracted": self.is_distracted,
            "is_fatigued": self.is_fatigued,
            "eyes_closed": self.eyes_closed,
            "shows_tension": self.shows_tension,
            "brow_raised": self.brow_raised,
            "head_down": self.head_down,
            "head_up": self.head_up,
            "is_speaking": self.is_speaking,
        }
    
    def to_fol_predicates(self) -> list:
        """
        Convert to FOL predicate strings for knowledge base.
        
        Returns list of true predicates in FOL notation.
        """
        predicates = []
        mapping = {
            "is_present": "Present(user)",
            "is_engaged": "Engaged(user)",
            "is_looking_away": "LookingAway(user)",
            "is_attentive": "Attentive(user)",
            "is_distracted": "Distracted(user)",
            "is_fatigued": "Fatigued(user)",
            "eyes_closed": "EyesClosed(user)",
            "shows_tension": "ShowsTension(user)",
            "brow_raised": "BrowRaised(user)",
            "head_down": "HeadDown(user)",
            "head_up": "HeadUp(user)",
            "is_speaking": "Speaking(user)",
        }
        
        for attr, fol in mapping.items():
            if getattr(self, attr, False):
                predicates.append(fol)
        
        return predicates


# =============================================================================
# CUE INTERPRETER
# =============================================================================

class CueInterpreter:
    """
    Interprets raw affective cues as symbolic predicates.
    
    This class bridges the CV module (continuous measurements) to the
    reasoning module (discrete symbolic logic).
    
    Academic Mapping:
    - Input: Sensor data (continuous values)
    - Output: Percept symbols (discrete predicates)
    - Function: Perception → Symbol grounding
    """
    
    def __init__(self, thresholds: CueThresholds = None):
        """
        Initialize the interpreter.
        
        Args:
            thresholds: Custom thresholds, or use defaults
        """
        self.thresholds = thresholds or CueThresholds()
        print("[CueInterpreter] Initialized with thresholds")
    
    def interpret(self, cues: AffectiveCues) -> SymbolicPredicates:
        """
        Convert raw cues to symbolic predicates.
        
        Args:
            cues: Raw affective cues from CVProcessor
            
        Returns:
            SymbolicPredicates with all derived boolean values
        """
        predicates = SymbolicPredicates()
        t = self.thresholds  # Shorthand
        
        # === PRESENCE ===
        predicates.is_present = cues.face_detected
        
        if not cues.face_detected:
            predicates.engagement_level = "none"
            return predicates
        
        # === HEAD POSE ===
        predicates.is_looking_away = abs(cues.head_yaw) > t.yaw_looking_away
        predicates.head_down = cues.head_pitch < t.pitch_head_down
        predicates.head_up = cues.head_pitch > t.pitch_head_up
        
        # === GAZE ===
        gaze_centered = t.gaze_centered_min <= cues.gaze_ratio <= t.gaze_centered_max
        
        # === EYES ===
        ear = cues.avg_eye_aspect_ratio
        predicates.eyes_closed = ear < t.eye_ar_closed
        predicates.is_fatigued = ear < t.eye_ar_fatigued and not predicates.eyes_closed
        eyes_wide = ear > t.eye_ar_wide
        
        # === MOUTH ===
        predicates.is_speaking = cues.mouth_aspect_ratio > t.mouth_open_threshold
        mouth_tense = cues.mouth_aspect_ratio < t.mouth_tense_threshold
        
        # === BROW ===
        predicates.brow_raised = cues.brow_height > t.brow_raised_threshold
        brow_furrowed = cues.brow_height < t.brow_furrowed_threshold
        
        # === COMPOSITE SIGNALS ===
        # Engagement: face present + looking at camera + eyes open
        predicates.is_engaged = (
            predicates.is_present and 
            not predicates.is_looking_away and 
            gaze_centered and
            not predicates.eyes_closed
        )
        
        # Attentive: engaged + alert eyes
        predicates.is_attentive = predicates.is_engaged and not predicates.is_fatigued
        
        # Distracted: looking away or gaze off
        predicates.is_distracted = predicates.is_looking_away or not gaze_centered
        
        # Tension: furrowed brow + tense mouth
        predicates.shows_tension = brow_furrowed and mouth_tense
        
        # === ENGAGEMENT LEVEL ===
        if predicates.is_attentive:
            predicates.engagement_level = "high"
        elif predicates.is_engaged:
            predicates.engagement_level = "medium"
        elif predicates.is_present:
            predicates.engagement_level = "low"
        else:
            predicates.engagement_level = "none"
        
        return predicates
    
    def get_summary(self, predicates: SymbolicPredicates) -> str:
        """Get human-readable summary of predicates."""
        if not predicates.is_present:
            return "No face detected"
        
        signals = []
        
        if predicates.is_attentive:
            signals.append("attentive")
        elif predicates.is_engaged:
            signals.append("engaged")
        elif predicates.is_distracted:
            signals.append("distracted")
        
        if predicates.is_fatigued:
            signals.append("fatigued")
        if predicates.shows_tension:
            signals.append("tense")
        if predicates.head_down:
            signals.append("head down")
        if predicates.is_speaking:
            signals.append("speaking")
        
        if not signals:
            signals.append("neutral")
        
        return f"Engagement: {predicates.engagement_level} | Signals: {', '.join(signals)}"
