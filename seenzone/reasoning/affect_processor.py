"""
Affect Processor - Continuous Affect Model
===========================================
Implements continuous affect detection using valence-arousal-engagement model.

Core Principle:
    "Track how the user feels over time, don't guess what they feel right now."

Pipeline:
    CV Raw Signals → Continuous Affect Variables → EMA Smoothing → Region-Based Mapping

Key Design:
    - Emotions are continuous signals, not frame-by-frame states
    - Discrete labels are derived summaries with confidence
    - Neutral = uncertainty region (no dominant affect), not a destination
    - Confidence decays over time when signals weaken
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import math


# =============================================================================
# EMOTION LABELS (Derived Summaries)
# =============================================================================

class AffectLabel(Enum):
    """
    Discrete emotion labels derived from continuous affect space.
    
    These are SUMMARIES, not primary detectors.
    """
    UNCERTAIN = "uncertain"      # No dominant region (replaces "Neutral")
    POSITIVE = "positive"        # High valence, moderate+ arousal
    SAD = "sad"                  # Low valence, low arousal
    STRESSED = "stressed"        # Negative valence, high arousal
    DISTRESSED = "distressed"    # Low valence, low engagement
    
    def __str__(self) -> str:
        return self.value.capitalize()


# =============================================================================
# AFFECT VARIABLES
# =============================================================================

@dataclass
class AffectVariables:
    """
    Continuous affect state at a point in time.
    
    These are the primary representation - discrete labels are derived from these.
    """
    # Core continuous variables
    valence: float = 0.0           # [-1.0, +1.0] negative ↔ positive
    arousal: float = 0.5           # [0.0, 1.0] calm ↔ activated
    engagement: float = 0.5        # [0.0, 1.0] disengaged ↔ attentive
    
    # Derived label and confidence
    label: AffectLabel = AffectLabel.UNCERTAIN
    confidence: float = 0.0        # [0.0, 1.0] how certain we are
    
    # Timestamps for decay
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "engagement": round(self.engagement, 3),
            "label": str(self.label),
            "confidence": round(self.confidence, 3),
        }


# =============================================================================
# AFFECT PROCESSOR
# =============================================================================

class AffectProcessor:
    """
    Processes raw CV signals into continuous affect variables with smoothing.
    
    Pipeline:
        1. Compute raw affect from CV signals (per frame)
        2. Apply EMA smoothing (prevents oscillation)
        3. Map to emotion regions (at low frequency)
        4. Compute confidence with decay
    """
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # EMA smoothing factor (0.05-0.2, lower = more smoothing)
    SMOOTHING_ALPHA = 0.12
    
    # Weights for valence computation
    VALENCE_WEIGHTS = {
        'smile_ratio': 1.5,      # Strong positive indicator
        'cheek_raise': 1.2,      # Duchenne smile (genuine)
        'expressive_face': 0.3,  # General expressiveness
        'tension': -0.8,         # Negative contribution
    }
    
    # Weights for arousal computation  
    AROUSAL_WEIGHTS = {
        'motion_energy': 1.0,
        'blink_rate': 0.3,       # Moderate contribution
        'mouth_activity': 0.5,
    }
    
    # Weights for engagement computation
    ENGAGEMENT_WEIGHTS = {
        'gaze_stability': 1.0,
        'presence': 1.5,         # Face detected is strong signal
        'looking_away': -1.0,    # Negative contribution
    }
    
    # Region thresholds for emotion mapping
    REGIONS = {
        'positive': {'valence_min': 0.4, 'arousal_min': 0.3},
        'sad': {'valence_max': -0.3, 'arousal_max': 0.3},
        'stressed': {'valence_max': 0.0, 'arousal_min': 0.7},
        'distressed': {'valence_max': -0.2, 'engagement_max': 0.3},
    }
    
    # Confidence decay rate (per second when signals weaken)
    CONFIDENCE_DECAY_RATE = 0.15
    
    # Minimum label change interval (seconds) - runs at ≤1Hz
    MIN_LABEL_INTERVAL = 1.0
    
    def __init__(self, smoothing_alpha: float = None):
        """
        Initialize the affect processor.
        
        Args:
            smoothing_alpha: Optional custom smoothing factor (0.05-0.2)
        """
        if smoothing_alpha is not None:
            self.SMOOTHING_ALPHA = max(0.05, min(0.2, smoothing_alpha))
        
        # Current smoothed state
        self._smoothed = AffectVariables()
        
        # Tracking for label updates
        self._last_label_update = 0.0
        self._last_update_time = time.time()
        
        # Logging counter
        self._log_counter = 0
        
        print(f"[AffectProcessor] Initialized (α={self.SMOOTHING_ALPHA})")
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process(self, cues) -> AffectVariables:
        """
        Process raw CV cues into continuous affect state.
        
        Args:
            cues: AffectiveCues from CVProcessor
            
        Returns:
            Current smoothed AffectVariables with label
        """
        current_time = time.time()
        self._log_counter += 1
        should_log = self._log_counter % 30 == 0
        
        # Step 1: Compute raw affect from CV signals
        raw_valence, raw_arousal, raw_engagement = self._compute_raw_affect(cues)
        
        if should_log:
            print(f"[AFFECT_RAW] valence={raw_valence:+.2f} "
                  f"arousal={raw_arousal:.2f} engagement={raw_engagement:.2f}")
        
        # Step 2: Apply EMA smoothing
        self._apply_smoothing(raw_valence, raw_arousal, raw_engagement, cues.face_detected)
        
        if should_log:
            print(f"[AFFECT_SMOOTH] valence={self._smoothed.valence:+.2f} "
                  f"arousal={self._smoothed.arousal:.2f} "
                  f"engagement={self._smoothed.engagement:.2f}")
        
        # Step 3: Update label at low frequency (≤1 Hz)
        if current_time - self._last_label_update >= self.MIN_LABEL_INTERVAL:
            self._update_label()
            self._last_label_update = current_time
        
        # Step 4: Apply confidence decay if signals are weak
        self._apply_confidence_decay(cues.face_detected, current_time)
        
        if should_log:
            print(f"[AFFECT_LABEL] state={self._smoothed.label} "
                  f"(confidence={self._smoothed.confidence:.2f})")
        
        self._smoothed.timestamp = current_time
        self._last_update_time = current_time
        
        return self._smoothed
    
    # =========================================================================
    # RAW AFFECT COMPUTATION
    # =========================================================================
    
    def _compute_raw_affect(self, cues) -> Tuple[float, float, float]:
        """
        Compute raw valence/arousal/engagement from CV signals.
        
        No smoothing at this stage - pure signal-to-affect mapping.
        """
        if not cues.face_detected:
            return 0.0, 0.0, 0.0
        
        # ----- VALENCE -----
        # Positive indicators
        smile_contrib = cues.smile_ratio * self.VALENCE_WEIGHTS['smile_ratio']
        cheek_contrib = cues.cheek_raise * self.VALENCE_WEIGHTS['cheek_raise']
        express_contrib = (1.0 if cues.mouth_activity > 0.15 else 0.0) * self.VALENCE_WEIGHTS['expressive_face']
        
        # Tension as negative (approximate from brow height)
        tension = max(0, 0.5 - cues.brow_height) * 2  # Furrowed = high tension
        tension_contrib = tension * self.VALENCE_WEIGHTS['tension']
        
        raw_valence = smile_contrib + cheek_contrib + express_contrib + tension_contrib
        # Normalize to [-1, +1]
        raw_valence = max(-1.0, min(1.0, raw_valence - 0.5))
        
        # ----- AROUSAL -----
        motion_contrib = cues.motion_energy * self.AROUSAL_WEIGHTS['motion_energy']
        blink_contrib = min(1.0, cues.blink_rate / 0.5) * self.AROUSAL_WEIGHTS['blink_rate']
        mouth_contrib = cues.mouth_activity * self.AROUSAL_WEIGHTS['mouth_activity']
        
        raw_arousal = motion_contrib + blink_contrib + mouth_contrib
        # Normalize to [0, 1]
        raw_arousal = max(0.0, min(1.0, raw_arousal))
        
        # ----- ENGAGEMENT -----
        gaze_contrib = cues.gaze_stability * self.ENGAGEMENT_WEIGHTS['gaze_stability']
        presence_contrib = 1.0 * self.ENGAGEMENT_WEIGHTS['presence']  # Always 1 if face detected
        
        # Looking away check (approximate from gaze ratio extremes)
        looking_away = 1.0 if (cues.gaze_ratio < 0.2 or cues.gaze_ratio > 0.8) else 0.0
        looking_contrib = looking_away * self.ENGAGEMENT_WEIGHTS['looking_away']
        
        raw_engagement = gaze_contrib + presence_contrib + looking_contrib
        # Normalize to [0, 1]
        raw_engagement = max(0.0, min(1.0, raw_engagement / 2.0))
        
        return raw_valence, raw_arousal, raw_engagement
    
    # =========================================================================
    # EMA SMOOTHING
    # =========================================================================
    
    def _apply_smoothing(self, raw_v: float, raw_a: float, raw_e: float, 
                         face_detected: bool) -> None:
        """
        Apply exponential moving average smoothing.
        
        x_t = α * x_raw + (1 - α) * x_(t-1)
        
        This prevents frame-to-frame oscillation and enforces emotional inertia.
        """
        α = self.SMOOTHING_ALPHA
        
        if not face_detected:
            # Decay toward neutral when no face
            self._smoothed.valence *= (1 - α * 0.5)
            self._smoothed.arousal = α * 0.5 + (1 - α) * self._smoothed.arousal
            self._smoothed.engagement *= (1 - α)
        else:
            self._smoothed.valence = α * raw_v + (1 - α) * self._smoothed.valence
            self._smoothed.arousal = α * raw_a + (1 - α) * self._smoothed.arousal
            self._smoothed.engagement = α * raw_e + (1 - α) * self._smoothed.engagement
    
    # =========================================================================
    # REGION-BASED LABEL MAPPING
    # =========================================================================
    
    def _update_label(self) -> None:
        """
        Map continuous affect to discrete emotion region.
        
        Uses closest-region selection, not priority-based.
        Neutral/Uncertain is the default when no region is dominant.
        """
        v = self._smoothed.valence
        a = self._smoothed.arousal
        e = self._smoothed.engagement
        
        # Calculate distance to each region
        region_scores = {}
        
        # Positive: valence > 0.4 AND arousal > 0.3
        if v > 0.2:  # Only consider if vaguely positive
            pos_v_score = max(0, (v - 0.4) / 0.6)  # How far above threshold
            pos_a_score = max(0, (a - 0.3) / 0.7)
            region_scores['positive'] = (pos_v_score + pos_a_score) / 2
        
        # Sad: valence < -0.3 AND arousal < 0.3
        if v < 0:  # Only consider if negative
            sad_v_score = max(0, (-0.3 - v) / 0.7)
            sad_a_score = max(0, (0.3 - a) / 0.3)
            region_scores['sad'] = (sad_v_score + sad_a_score) / 2
        
        # Stressed: valence < 0 AND arousal > 0.7
        if v < 0.1 and a > 0.5:
            stress_v_score = max(0, (0.0 - v) / 1.0)
            stress_a_score = max(0, (a - 0.7) / 0.3)
            region_scores['stressed'] = (stress_v_score + stress_a_score) / 2
        
        # Distressed: valence < -0.2 AND engagement < 0.3
        if v < 0 and e < 0.5:
            dist_v_score = max(0, (-0.2 - v) / 0.8)
            dist_e_score = max(0, (0.3 - e) / 0.3)
            region_scores['distressed'] = (dist_v_score + dist_e_score) / 2
        
        # Find best matching region
        if region_scores:
            best_region = max(region_scores, key=region_scores.get)
            best_score = region_scores[best_region]
            
            # Only assign label if score is meaningful (> 0.2)
            if best_score > 0.2:
                label_map = {
                    'positive': AffectLabel.POSITIVE,
                    'sad': AffectLabel.SAD,
                    'stressed': AffectLabel.STRESSED,
                    'distressed': AffectLabel.DISTRESSED,
                }
                self._smoothed.label = label_map[best_region]
                # Confidence based on how well we match the region
                self._smoothed.confidence = min(1.0, best_score + 0.3)
                return
        
        # Default: Uncertain (no dominant region)
        self._smoothed.label = AffectLabel.UNCERTAIN
        # Low confidence when uncertain
        self._smoothed.confidence = max(0.1, 0.3 - abs(v) * 0.5)
    
    # =========================================================================
    # CONFIDENCE DECAY
    # =========================================================================
    
    def _apply_confidence_decay(self, face_detected: bool, current_time: float) -> None:
        """
        Decay confidence over time when signals weaken.
        
        Prevents "I smiled once so I'm Positive forever".
        """
        dt = current_time - self._last_update_time
        
        if not face_detected:
            # Faster decay when no face
            self._smoothed.confidence *= math.exp(-self.CONFIDENCE_DECAY_RATE * 2 * dt)
        elif self._smoothed.label == AffectLabel.UNCERTAIN:
            # Moderate decay in uncertain state
            self._smoothed.confidence *= math.exp(-self.CONFIDENCE_DECAY_RATE * dt)
        
        # Ensure minimum
        self._smoothed.confidence = max(0.0, self._smoothed.confidence)
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_current_state(self) -> AffectVariables:
        """Get the current smoothed affect state."""
        return self._smoothed
    
    def get_emotion_label(self) -> str:
        """Get the current emotion label as string."""
        return str(self._smoothed.label)
    
    def get_emotional_state_name(self) -> str:
        """
        Map affect label to EmotionalState name for compatibility.
        
        Returns a string matching EmotionalState enum names.
        """
        mapping = {
            AffectLabel.UNCERTAIN: "S0_NEUTRAL",
            AffectLabel.POSITIVE: "S5_POSITIVE_STATE",
            AffectLabel.SAD: "S1_SADNESS_DETECTED",
            AffectLabel.STRESSED: "S3_HIGH_STRESS",
            AffectLabel.DISTRESSED: "S6_DISTRESSED_SILENT",
        }
        return mapping.get(self._smoothed.label, "S0_NEUTRAL")
    
    def reset(self) -> None:
        """Reset to initial uncertain state."""
        self._smoothed = AffectVariables()
        self._last_label_update = 0.0
        self._log_counter = 0
        print("[AffectProcessor] Reset")
