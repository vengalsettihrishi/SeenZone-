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
    
    NEW: Relative thresholds use delta from calibrated baseline,
    eliminating laptop camera position bias.
    """
    # Head pose thresholds (ABSOLUTE - legacy, kept for fallback)
    yaw_looking_away: float = 0.25      # Absolute yaw > this = looking away
    pitch_head_down: float = -0.15      # Pitch < this = head down (DEPRECATED)
    pitch_head_up: float = 0.15         # Pitch > this = head up
    
    # NEW: Relative head pose thresholds (delta from baseline)
    pitch_lowered_delta: float = 0.20   # Pitch < baseline - this = significantly lowered
    pitch_raised_delta: float = 0.15    # Pitch > baseline + this = raised
    
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
    
    # NEW: Expressiveness thresholds
    smile_ratio_threshold: float = 0.55   # Above = smiling
    cheek_raise_threshold: float = 0.45   # Above = cheek raised (genuine smile)
    mouth_activity_threshold: float = 0.15  # Above = expressive face
    
    # NEW: Engagement thresholds (non-posture-based)
    gaze_stability_threshold: float = 0.70  # Above = stable attentive gaze
    motion_energy_low: float = 0.10         # Below = low energy (sadness indicator)
    motion_energy_high: float = 0.35        # Above = active


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
    
    NEW: Added relative predicates and expressiveness signals
    for laptop-aware, multi-signal affect detection.
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
    
    # Head position (LEGACY - absolute thresholds)
    head_down: bool = False             # Looking down (DEPRECATED - use head_lowered_significantly)
    head_up: bool = False               # Looking up
    
    # NEW: Relative head position (laptop-aware)
    head_lowered_significantly: bool = False   # Head lowered RELATIVE TO BASELINE
    head_raised_significantly: bool = False    # Head raised RELATIVE TO BASELINE
    
    # NEW: Positive expressiveness signals
    is_smiling: bool = False            # Mouth corners raised
    expressive_face: bool = False       # Mouth activity or cheek raise
    cheek_raised: bool = False          # Genuine smile indicator
    
    # NEW: Non-posture engagement signals
    is_active: bool = False             # High motion energy
    gaze_stable: bool = False           # Stable attentive gaze
    low_motion_energy: bool = False     # Low expressiveness (sadness indicator)
    
    # Verbal signals
    is_speaking: bool = False           # Mouth moving/open
    
    # Overall engagement level (composite)
    engagement_level: str = "unknown"   # "high", "medium", "low", "none"
    
    # NEW: Sadness evidence tracking (for multi-signal detection)
    sadness_evidence_count: int = 0     # How many sadness indicators are present
    
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
            # NEW predicates
            "head_lowered_significantly": self.head_lowered_significantly,
            "head_raised_significantly": self.head_raised_significantly,
            "is_smiling": self.is_smiling,
            "expressive_face": self.expressive_face,
            "cheek_raised": self.cheek_raised,
            "is_active": self.is_active,
            "gaze_stable": self.gaze_stable,
            "low_motion_energy": self.low_motion_energy,
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
            # NEW predicates
            "head_lowered_significantly": "HeadLoweredSignificantly(user)",
            "head_raised_significantly": "HeadRaisedSignificantly(user)",
            "is_smiling": "Smiling(user)",
            "expressive_face": "ExpressiveFace(user)",
            "cheek_raised": "CheekRaised(user)",
            "is_active": "Active(user)",
            "gaze_stable": "GazeStable(user)",
            "low_motion_energy": "LowMotionEnergy(user)",
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
    
    NEW: Supports baseline-relative predicates for laptop-aware detection.
    """
    
    def __init__(self, thresholds: CueThresholds = None):
        """
        Initialize the interpreter.
        
        Args:
            thresholds: Custom thresholds, or use defaults
        """
        self.thresholds = thresholds or CueThresholds()
        self.baseline: dict = None  # Will be set by CVProcessor
        self._log_counter = 0
        print("[CueInterpreter] Initialized with thresholds")
    
    def set_baseline(self, baseline: dict) -> None:
        """
        Set calibrated baseline values for relative detection.
        
        Args:
            baseline: Dict with 'head_pitch', 'head_yaw', 'gaze_ratio'
        """
        self.baseline = baseline
        print(f"[CueInterpreter] Baseline set: pitch={baseline.get('head_pitch', 0):.3f}")
    
    def interpret(self, cues: AffectiveCues, baseline_delta: dict = None) -> SymbolicPredicates:
        """
        Convert raw cues to symbolic predicates.
        
        Args:
            cues: Raw affective cues from CVProcessor
            baseline_delta: Optional dict with pitch_delta, yaw_delta, gaze_delta
            
        Returns:
            SymbolicPredicates with all derived boolean values
        """
        predicates = SymbolicPredicates()
        t = self.thresholds  # Shorthand
        
        self._log_counter += 1
        should_log = self._log_counter % 30 == 0
        
        # === PRESENCE ===
        predicates.is_present = cues.face_detected
        
        if not cues.face_detected:
            if should_log:
                print("[CV_RAW] No face detected - check webcam and lighting")
            predicates.engagement_level = "none"
            return predicates
        
        # === RAW CV LOGGING ===
        if should_log:
            print(f"[CV_RAW] face=True pitch={cues.head_pitch:+.3f} yaw={cues.head_yaw:+.3f} "
                  f"smile_ratio={cues.smile_ratio:.2f}")
        
        # === BASELINE DELTA LOGGING ===
        pitch_delta = 0.0
        if baseline_delta:
            pitch_delta = baseline_delta.get('pitch_delta', 0.0)
            if should_log:
                lowered = pitch_delta < -t.pitch_lowered_delta
                print(f"[BASELINE_DELTA] pitch_delta={pitch_delta:+.3f} "
                      f"→ {'LOWERED significantly' if lowered else 'NOT lowered'}")
        
        # === HEAD POSE (RELATIVE - LAPTOP AWARE) ===
        predicates.is_looking_away = abs(cues.head_yaw) > t.yaw_looking_away
        
        # NEW: Use RELATIVE pitch from baseline (eliminates laptop bias)
        if baseline_delta:
            predicates.head_lowered_significantly = pitch_delta < -t.pitch_lowered_delta
            predicates.head_raised_significantly = pitch_delta > t.pitch_raised_delta
        else:
            # Fallback to absolute thresholds if no baseline
            predicates.head_lowered_significantly = False  # Don't trigger without baseline
            predicates.head_raised_significantly = cues.head_pitch > t.pitch_head_up
        
        # LEGACY: Keep absolute head_down for backward compatibility (but prefer relative)
        predicates.head_down = cues.head_pitch < t.pitch_head_down
        predicates.head_up = cues.head_pitch > t.pitch_head_up
        
        # === GAZE ===
        gaze_centered = t.gaze_centered_min <= cues.gaze_ratio <= t.gaze_centered_max
        predicates.gaze_stable = cues.gaze_stability > t.gaze_stability_threshold
        
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
        
        # === NEW: EXPRESSIVENESS SIGNALS ===
        predicates.is_smiling = cues.smile_ratio > t.smile_ratio_threshold
        predicates.cheek_raised = cues.cheek_raise > t.cheek_raise_threshold
        predicates.expressive_face = (
            cues.mouth_activity > t.mouth_activity_threshold or
            predicates.cheek_raised
        )
        
        # === NEW: ENGAGEMENT SIGNALS (NON-POSTURE BASED) ===
        predicates.is_active = cues.motion_energy > t.motion_energy_high
        predicates.low_motion_energy = cues.motion_energy < t.motion_energy_low
        
        # === LOGGING: FACE EXPRESSIVENESS ===
        if should_log:
            print(f"[CV_FACE] smile_ratio={cues.smile_ratio:.2f} "
                  f"cheek_raise={cues.cheek_raise:.2f} "
                  f"mouth_activity={cues.mouth_activity:.2f}")
        
        # === LOGGING: ATTENTION ===
        if should_log:
            print(f"[CV_ATTENTION] gaze_stability={cues.gaze_stability:.2f} "
                  f"blink_rate={cues.blink_rate:.2f} "
                  f"motion_energy={cues.motion_energy:.2f}")
        
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
        
        # === NEW: SADNESS EVIDENCE COUNTING (MULTI-SIGNAL) ===
        sadness_signals = 0
        if predicates.head_lowered_significantly:
            sadness_signals += 1
        if predicates.low_motion_energy:
            sadness_signals += 1
        if not predicates.expressive_face and not predicates.is_smiling:
            sadness_signals += 1
        
        predicates.sadness_evidence_count = sadness_signals
        
        if should_log:
            print(f"[SADNESS_EVIDENCE] signals={sadness_signals}/3 "
                  f"→ {'BLOCK sadness' if sadness_signals < 2 else 'allow sadness'}")
        
        # === LOGGING: CUES SUMMARY ===
        if should_log:
            cues_list = []
            if predicates.head_lowered_significantly:
                cues_list.append("HeadLoweredSignificantly")
            if predicates.is_smiling:
                cues_list.append("Smiling")
            if predicates.expressive_face:
                cues_list.append("ExpressiveFace")
            if predicates.is_active:
                cues_list.append("Active")
            if predicates.low_motion_energy:
                cues_list.append("LowMotionEnergy")
            if predicates.gaze_stable:
                cues_list.append("GazeStable")
            print(f"[CUES] {', '.join(cues_list) if cues_list else 'None'}")
        
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
        
        # NEW: Positive signals
        if predicates.is_smiling:
            signals.append("smiling")
        if predicates.is_active:
            signals.append("active")
        
        # Negative signals
        if predicates.is_fatigued:
            signals.append("fatigued")
        if predicates.shows_tension:
            signals.append("tense")
        if predicates.head_lowered_significantly:
            signals.append("head lowered")
        if predicates.low_motion_energy:
            signals.append("low energy")
        if predicates.is_speaking:
            signals.append("speaking")
        
        if not signals:
            signals.append("neutral")
        
        return f"Engagement: {predicates.engagement_level} | Signals: {', '.join(signals)}"
