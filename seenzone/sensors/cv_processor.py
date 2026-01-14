"""
CV Processor - Affective Cue Extraction
========================================
Extracts non-verbal affective cues from webcam frames using MediaPipe Face Mesh.

PEAS Mapping: Part of SENSOR component (visual processing pipeline)

This module does NOT classify emotions. It extracts observable physical signals:
- Head pose (pitch, yaw, roll)
- Eye aspect ratio (openness)
- Gaze direction
- Facial landmark positions

These raw measurements are converted to symbolic cues by the CueInterpreter.

Privacy: All processing is LOCAL. Frames are never transmitted externally.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import deque
import cv2
import time

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[CVProcessor] Warning: MediaPipe not installed. CV features disabled.")


# =============================================================================
# AFFECTIVE CUES DATA STRUCTURE
# =============================================================================

@dataclass
class AffectiveCues:
    """
    Container for extracted affective cues from a single frame.
    
    These are raw measurements, not interpretations. The CueInterpreter
    converts these to symbolic predicates for the reasoning module.
    
    Attributes:
        face_detected: Whether a face was found in the frame
        face_confidence: Detection confidence (0.0 to 1.0)
        head_pitch: Vertical head angle (-1=down, 0=level, 1=up)
        head_yaw: Horizontal head angle (-1=left, 0=center, 1=right)
        head_roll: Head tilt angle (-1=left tilt, 0=level, 1=right tilt)
        left_eye_aspect_ratio: Left eye openness (0=closed, ~0.3=open)
        right_eye_aspect_ratio: Right eye openness
        gaze_ratio: Horizontal gaze position (0=left, 0.5=center, 1=right)
        mouth_aspect_ratio: Mouth openness (0=closed, higher=open)
        brow_height: Normalized eyebrow height (higher=raised)
        
        # NEW: Expressiveness variables
        smile_ratio: Lip corner distance / face width (higher = smiling)
        mouth_activity: Temporal change in mouth landmarks
        cheek_raise: Cheek-to-eye distance change (genuine smile indicator)
        
        # NEW: Attention/engagement variables
        gaze_stability: Variance of gaze over N frames (higher = more stable)
        blink_rate: Blinks detected per second
        motion_energy: Overall landmark movement over time
    """
    face_detected: bool = False
    face_confidence: float = 0.0
    
    # Head pose
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    
    # Eye metrics
    left_eye_aspect_ratio: float = 0.0
    right_eye_aspect_ratio: float = 0.0
    
    # Gaze
    gaze_ratio: float = 0.5  # 0.5 = centered
    
    # Mouth
    mouth_aspect_ratio: float = 0.0
    
    # Brow
    brow_height: float = 0.0
    
    # NEW: Expressiveness variables
    smile_ratio: float = 0.0           # Lip corner distance / face width
    mouth_activity: float = 0.0        # Temporal change in mouth landmarks
    cheek_raise: float = 0.0           # Cheek-to-eye distance change
    
    # NEW: Attention/engagement variables
    gaze_stability: float = 0.5        # Variance of gaze over N frames (higher = stable)
    blink_rate: float = 0.0            # Blinks detected per second
    motion_energy: float = 0.0         # Overall landmark movement over time
    
    @property
    def avg_eye_aspect_ratio(self) -> float:
        """Average eye aspect ratio across both eyes."""
        return (self.left_eye_aspect_ratio + self.right_eye_aspect_ratio) / 2
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/debugging."""
        return {
            "face_detected": self.face_detected,
            "face_confidence": round(self.face_confidence, 3),
            "head_pitch": round(self.head_pitch, 3),
            "head_yaw": round(self.head_yaw, 3),
            "head_roll": round(self.head_roll, 3),
            "eye_aspect_ratio": round(self.avg_eye_aspect_ratio, 3),
            "gaze_ratio": round(self.gaze_ratio, 3),
            "mouth_aspect_ratio": round(self.mouth_aspect_ratio, 3),
            "brow_height": round(self.brow_height, 3),
            # NEW: Expressiveness
            "smile_ratio": round(self.smile_ratio, 3),
            "mouth_activity": round(self.mouth_activity, 3),
            "cheek_raise": round(self.cheek_raise, 3),
            # NEW: Attention
            "gaze_stability": round(self.gaze_stability, 3),
            "blink_rate": round(self.blink_rate, 3),
            "motion_energy": round(self.motion_energy, 3),
        }



# =============================================================================
# LANDMARK INDICES (MediaPipe Face Mesh)
# =============================================================================
# MediaPipe Face Mesh provides 468 landmarks. These are key indices we use.

class LandmarkIndices:
    """Key landmark indices for MediaPipe Face Mesh (468 points)."""
    
    # Left eye landmarks
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_EYE_OUTER = 263
    LEFT_EYE_INNER = 362
    LEFT_EYE_TOP = 386
    LEFT_EYE_BOTTOM = 374
    
    # Right eye landmarks
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 133
    RIGHT_EYE_TOP = 159
    RIGHT_EYE_BOTTOM = 145
    
    # Iris landmarks (for gaze)
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    
    # Mouth landmarks
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    
    # Eyebrow landmarks
    LEFT_BROW = [66, 105, 63, 70]
    RIGHT_BROW = [296, 334, 293, 300]
    
    # Nose tip (for head pose reference)
    NOSE_TIP = 1
    
    # Face oval points (for head pose)
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


# =============================================================================
# BASELINE CALIBRATOR (LAPTOP-AWARE)
# =============================================================================

class BaselineCalibrator:
    """
    Captures baseline head pose during first ~2 seconds of operation.
    
    This eliminates the laptop camera bias where head pitch is always
    slightly negative due to camera position below eye level.
    
    All subsequent measurements are compared relative to baseline.
    """
    
    def __init__(self, calibration_frames: int = 60):
        """
        Initialize calibrator.
        
        Args:
            calibration_frames: Number of frames to collect (~2 seconds at 30fps)
        """
        self.calibration_frames = calibration_frames
        self.samples: List[Dict[str, float]] = []
        self.baseline: Optional[Dict[str, float]] = None
        self.is_calibrated: bool = False
        print(f"[BaselineCalibrator] Initialized (calibration_frames={calibration_frames})")
    
    def add_sample(self, cues: 'AffectiveCues') -> bool:
        """
        Add a sample during calibration phase.
        
        Args:
            cues: Current affective cues
            
        Returns:
            True if calibration just completed, False otherwise
        """
        if self.is_calibrated:
            return False
        
        if not cues.face_detected:
            return False
        
        self.samples.append({
            'head_pitch': cues.head_pitch,
            'head_yaw': cues.head_yaw,
            'gaze_ratio': cues.gaze_ratio,
        })
        
        if len(self.samples) >= self.calibration_frames:
            self._finalize_calibration()
            return True
        
        return False
    
    def _finalize_calibration(self) -> None:
        """Calculate baseline from collected samples."""
        if not self.samples:
            return
        
        self.baseline = {
            'head_pitch': np.mean([s['head_pitch'] for s in self.samples]),
            'head_yaw': np.mean([s['head_yaw'] for s in self.samples]),
            'gaze_ratio': np.mean([s['gaze_ratio'] for s in self.samples]),
        }
        self.is_calibrated = True
        
        print(f"[BASELINE] Calibration complete:")
        print(f"[BASELINE] head_pitch={self.baseline['head_pitch']:.3f}")
        print(f"[BASELINE] head_yaw={self.baseline['head_yaw']:.3f}")
        print(f"[BASELINE] gaze={self.baseline['gaze_ratio']:.3f}")
    
    def get_baseline(self) -> Dict[str, float]:
        """Get baseline values (returns zeros if not calibrated)."""
        if self.baseline:
            return self.baseline.copy()
        return {'head_pitch': 0.0, 'head_yaw': 0.0, 'gaze_ratio': 0.5}
    
    def get_delta(self, cues: 'AffectiveCues') -> Dict[str, float]:
        """
        Get delta from baseline for current cues.
        
        Args:
            cues: Current affective cues
            
        Returns:
            Dictionary with delta values
        """
        baseline = self.get_baseline()
        return {
            'pitch_delta': cues.head_pitch - baseline['head_pitch'],
            'yaw_delta': cues.head_yaw - baseline['head_yaw'],
            'gaze_delta': cues.gaze_ratio - baseline['gaze_ratio'],
        }
    
    def reset(self) -> None:
        """Reset calibration."""
        self.samples.clear()
        self.baseline = None
        self.is_calibrated = False
        print("[BaselineCalibrator] Reset")


# =============================================================================
# TEMPORAL TRACKER
# =============================================================================

class TemporalTracker:
    """
    Tracks temporal dynamics across frames for attention/engagement signals.
    
    Provides:
    - Gaze stability (variance over time)
    - Blink detection and rate
    - Motion energy (overall movement)
    - Mouth activity
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize tracker.
        
        Args:
            window_size: Number of frames for rolling window (~1 second at 30fps)
        """
        self.window_size = window_size
        
        # Rolling windows for different metrics
        self.gaze_history: deque = deque(maxlen=window_size)
        self.ear_history: deque = deque(maxlen=window_size)
        self.mouth_history: deque = deque(maxlen=window_size)
        self.pose_history: deque = deque(maxlen=window_size)
        
        # Blink detection state
        self.blink_timestamps: deque = deque(maxlen=50)
        self.last_ear: float = 0.3
        self.in_blink: bool = False
        self.blink_ear_threshold: float = 0.20
        
        # Timing
        self.last_update_time: float = time.time()
        
        print(f"[TemporalTracker] Initialized (window_size={window_size})")
    
    def update(self, cues: 'AffectiveCues') -> None:
        """
        Update tracker with new frame data.
        
        Args:
            cues: Current affective cues
        """
        if not cues.face_detected:
            return
        
        current_time = time.time()
        
        # Update histories
        self.gaze_history.append(cues.gaze_ratio)
        self.ear_history.append(cues.avg_eye_aspect_ratio)
        self.mouth_history.append(cues.mouth_aspect_ratio)
        self.pose_history.append((cues.head_pitch, cues.head_yaw, cues.head_roll))
        
        # Blink detection
        ear = cues.avg_eye_aspect_ratio
        if not self.in_blink and ear < self.blink_ear_threshold:
            self.in_blink = True
        elif self.in_blink and ear > self.blink_ear_threshold + 0.05:
            self.in_blink = False
            self.blink_timestamps.append(current_time)
        
        self.last_ear = ear
        self.last_update_time = current_time
    
    def get_gaze_stability(self) -> float:
        """
        Calculate gaze stability (0 = unstable, 1 = very stable).
        
        Returns:
            Stability score based on inverse variance
        """
        if len(self.gaze_history) < 5:
            return 0.5
        
        variance = np.var(list(self.gaze_history))
        # Convert variance to 0-1 stability score (lower variance = higher stability)
        # Variance of 0 → stability 1.0, variance of 0.1+ → stability ~0
        stability = 1.0 / (1.0 + variance * 50)
        return min(1.0, max(0.0, stability))
    
    def get_blink_rate(self) -> float:
        """
        Calculate blinks per second over recent window.
        
        Returns:
            Blinks per second
        """
        if not self.blink_timestamps:
            return 0.0
        
        current_time = time.time()
        # Count blinks in last 10 seconds
        window_seconds = 10.0
        recent_blinks = sum(1 for t in self.blink_timestamps 
                           if current_time - t < window_seconds)
        
        return recent_blinks / window_seconds
    
    def get_motion_energy(self) -> float:
        """
        Calculate overall motion energy from pose changes.
        
        Returns:
            Motion energy score (0 = still, 1 = high movement)
        """
        if len(self.pose_history) < 5:
            return 0.0
        
        poses = list(self.pose_history)
        
        # Calculate frame-to-frame differences
        diffs = []
        for i in range(1, len(poses)):
            prev = poses[i-1]
            curr = poses[i]
            diff = sum((c - p) ** 2 for c, p in zip(curr, prev))
            diffs.append(diff)
        
        if not diffs:
            return 0.0
        
        # Average motion and scale to 0-1
        avg_motion = np.mean(diffs)
        # Scale: 0.01 motion → ~0.5 energy
        energy = min(1.0, avg_motion * 50)
        return energy
    
    def get_mouth_activity(self) -> float:
        """
        Calculate mouth movement activity.
        
        Returns:
            Activity score (0 = still, 1 = high activity)
        """
        if len(self.mouth_history) < 5:
            return 0.0
        
        variance = np.var(list(self.mouth_history))
        # Scale variance to 0-1 activity score
        activity = min(1.0, variance * 100)
        return activity
    
    def reset(self) -> None:
        """Reset all tracking."""
        self.gaze_history.clear()
        self.ear_history.clear()
        self.mouth_history.clear()
        self.pose_history.clear()
        self.blink_timestamps.clear()
        self.in_blink = False
        print("[TemporalTracker] Reset")


# =============================================================================
# CV PROCESSOR CLASS
# =============================================================================

class CVProcessor:
    """
    Computer Vision processor using MediaPipe Face Mesh.
    
    Extracts affective cues from video frames:
    - Face detection and confidence
    - Head pose estimation (pitch, yaw, roll)
    - Eye aspect ratio (for fatigue/alertness)
    - Gaze direction
    - Mouth state
    - Eyebrow position
    - NEW: Smile ratio, cheek raise (expressiveness)
    - NEW: Temporal dynamics via BaselineCalibrator and TemporalTracker
    
    All processing is performed locally. No data is transmitted externally.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 refine_landmarks: bool = True,
                 calibration_frames: int = 60):
        """
        Initialize the CV processor.
        
        Args:
            min_detection_confidence: Minimum face detection confidence
            min_tracking_confidence: Minimum tracking confidence
            refine_landmarks: Whether to refine eye/lip landmarks
            calibration_frames: Frames for baseline calibration (~2 seconds)
        """
        self.enabled = MEDIAPIPE_AVAILABLE
        
        if not self.enabled:
            print("[CVProcessor] Disabled: MediaPipe not available")
            return
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Single user assumption
            refine_landmarks=refine_landmarks,  # Enables iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # For visualization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # NEW: Baseline calibrator (laptop-aware)
        self.calibrator = BaselineCalibrator(calibration_frames=calibration_frames)
        
        # NEW: Temporal tracker for dynamics
        self.tracker = TemporalTracker(window_size=30)
        
        # Frame counter for logging
        self._frame_count = 0
        
        print(f"[CVProcessor] Initialized (detection={min_detection_confidence}, tracking={min_tracking_confidence})")
    
    def process(self, frame: np.ndarray) -> AffectiveCues:
        """
        Process a frame and extract affective cues.
        
        Args:
            frame: BGR image from OpenCV (numpy array)
            
        Returns:
            AffectiveCues containing all extracted measurements
        """
        cues = AffectiveCues()
        
        if not self.enabled or frame is None:
            return cues
        
        # Convert BGR to RGB for MediaPipe
        convert_start = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Performance optimization
        convert_time = (time.perf_counter() - convert_start) * 1000
        
        # Process with Face Mesh (this is the bottleneck)
        mp_start = time.perf_counter()
        results = self.face_mesh.process(rgb_frame)
        mp_time = (time.perf_counter() - mp_start) * 1000
        
        # Log MediaPipe timing periodically (every ~30 frames to reduce noise)
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            print(f"[TIMING] MediaPipe: {mp_time:.1f}ms (convert: {convert_time:.1f}ms)")
        
        if not results.multi_face_landmarks:
            return cues  # No face detected
        
        # Get first face (single user assumption)
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        # Face detected
        cues.face_detected = True
        cues.face_confidence = self._estimate_confidence(landmarks)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Extract basic cues
        cues.head_pitch, cues.head_yaw, cues.head_roll = self._estimate_head_pose(landmarks, w, h)
        cues.left_eye_aspect_ratio = self._calculate_eye_aspect_ratio(landmarks, LandmarkIndices.LEFT_EYE)
        cues.right_eye_aspect_ratio = self._calculate_eye_aspect_ratio(landmarks, LandmarkIndices.RIGHT_EYE)
        cues.gaze_ratio = self._calculate_gaze_ratio(landmarks, w)
        cues.mouth_aspect_ratio = self._calculate_mouth_aspect_ratio(landmarks)
        cues.brow_height = self._calculate_brow_height(landmarks)
        
        # NEW: Expressiveness cues
        cues.smile_ratio = self._calculate_smile_ratio(landmarks)
        cues.cheek_raise = self._calculate_cheek_raise(landmarks)
        
        # NEW: Baseline calibration (first ~2 seconds)
        if not self.calibrator.is_calibrated:
            just_calibrated = self.calibrator.add_sample(cues)
            if just_calibrated:
                print("[CVProcessor] Baseline calibration complete!")
        
        # NEW: Update temporal tracker
        self.tracker.update(cues)
        
        # NEW: Get temporal dynamics from tracker
        cues.gaze_stability = self.tracker.get_gaze_stability()
        cues.blink_rate = self.tracker.get_blink_rate()
        cues.motion_energy = self.tracker.get_motion_energy()
        cues.mouth_activity = self.tracker.get_mouth_activity()
        
        return cues
    
    def get_baseline(self) -> Dict[str, float]:
        """Get the calibrated baseline values."""
        return self.calibrator.get_baseline()
    
    def get_baseline_delta(self, cues: AffectiveCues) -> Dict[str, float]:
        """Get the delta from baseline for current cues."""
        return self.calibrator.get_delta(cues)
    
    def is_calibrated(self) -> bool:
        """Check if baseline calibration is complete."""
        return self.calibrator.is_calibrated
    
    def reset_calibration(self) -> None:
        """Reset baseline calibration."""
        self.calibrator.reset()
        self.tracker.reset()
    
    def _estimate_confidence(self, landmarks) -> float:
        """Estimate detection confidence from landmark visibility."""
        # Average visibility of key landmarks
        key_indices = [LandmarkIndices.NOSE_TIP, 
                       LandmarkIndices.LEFT_EYE_INNER, 
                       LandmarkIndices.RIGHT_EYE_INNER]
        visibilities = [landmarks[i].visibility for i in key_indices if hasattr(landmarks[i], 'visibility')]
        return sum(visibilities) / len(visibilities) if visibilities else 0.8
    
    def _estimate_head_pose(self, landmarks, img_w: int, img_h: int) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll) from facial landmarks.
        
        Uses key facial points to estimate 3D head orientation.
        Returns normalized values in range [-1, 1].
        """
        # Get key points for pose estimation
        nose = landmarks[LandmarkIndices.NOSE_TIP]
        left_eye = landmarks[LandmarkIndices.LEFT_EYE_OUTER]
        right_eye = landmarks[LandmarkIndices.RIGHT_EYE_OUTER]
        
        # Convert to pixel coordinates
        nose_x, nose_y = nose.x * img_w, nose.y * img_h
        left_eye_x, left_eye_y = left_eye.x * img_w, left_eye.y * img_h
        right_eye_x, right_eye_y = right_eye.x * img_w, right_eye.y * img_h
        
        # Yaw: horizontal rotation (looking left/right)
        # Based on nose position relative to eye midpoint
        eye_center_x = (left_eye_x + right_eye_x) / 2
        yaw = (nose_x - eye_center_x) / (img_w * 0.1)  # Normalize
        yaw = max(-1, min(1, yaw))  # Clamp
        
        # Pitch: vertical rotation (looking up/down)
        # Based on nose y position relative to eyes
        eye_center_y = (left_eye_y + right_eye_y) / 2
        pitch = (eye_center_y - nose_y) / (img_h * 0.1)  # Normalize
        pitch = max(-1, min(1, pitch))
        
        # Roll: head tilt
        # Based on angle between eyes
        eye_diff_y = right_eye_y - left_eye_y
        eye_diff_x = right_eye_x - left_eye_x
        roll = np.arctan2(eye_diff_y, eye_diff_x) / (np.pi / 4)  # Normalize
        roll = max(-1, min(1, roll))
        
        return pitch, yaw, roll
    
    def _calculate_eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink/fatigue detection.
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        Lower EAR = more closed eye
        """
        if len(eye_indices) < 6:
            return 0.3  # Default open value
        
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]
        
        # Vertical distances
        v1 = np.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2)
        v2 = np.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2)
        
        # Horizontal distance
        h = np.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2)
        
        if h == 0:
            return 0.3
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _calculate_gaze_ratio(self, landmarks, img_w: int) -> float:
        """
        Calculate horizontal gaze ratio.
        
        Returns 0.0 (looking left) to 1.0 (looking right), 0.5 = center
        """
        # Use iris center relative to eye corners
        left_iris = landmarks[LandmarkIndices.LEFT_IRIS[0]]
        left_inner = landmarks[LandmarkIndices.LEFT_EYE_INNER]
        left_outer = landmarks[LandmarkIndices.LEFT_EYE_OUTER]
        
        # Calculate where iris is between inner and outer corners
        eye_width = abs(left_outer.x - left_inner.x)
        if eye_width == 0:
            return 0.5
        
        iris_position = (left_iris.x - left_inner.x) / eye_width
        return max(0, min(1, iris_position))
    
    def _calculate_mouth_aspect_ratio(self, landmarks) -> float:
        """
        Calculate mouth aspect ratio (openness).
        
        Higher value = more open mouth
        """
        top = landmarks[LandmarkIndices.MOUTH_TOP]
        bottom = landmarks[LandmarkIndices.MOUTH_BOTTOM]
        left = landmarks[LandmarkIndices.MOUTH_LEFT]
        right = landmarks[LandmarkIndices.MOUTH_RIGHT]
        
        vertical = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
        horizontal = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
        
        if horizontal == 0:
            return 0.0
        
        return vertical / horizontal
    
    def _calculate_brow_height(self, landmarks) -> float:
        """
        Calculate normalized eyebrow height.
        
        Higher value = raised eyebrows (surprise/stress indicator)
        """
        # Average height of eyebrow points relative to eye
        left_brow_avg = sum(landmarks[i].y for i in LandmarkIndices.LEFT_BROW) / len(LandmarkIndices.LEFT_BROW)
        left_eye_top = landmarks[LandmarkIndices.LEFT_EYE_TOP].y
        
        # Distance from brow to eye (normalized, inverted so higher = raised)
        brow_distance = left_eye_top - left_brow_avg
        
        # Normalize to roughly 0-1 range
        return max(0, min(1, brow_distance * 10 + 0.5))
    
    def _calculate_smile_ratio(self, landmarks) -> float:
        """
        Calculate smile ratio based on mouth corner positions.
        
        Higher value = more of a smile (lip corners pulled up/out)
        """
        # Mouth corners
        left_corner = landmarks[LandmarkIndices.MOUTH_LEFT]
        right_corner = landmarks[LandmarkIndices.MOUTH_RIGHT]
        
        # Use face width for normalization (approximate with outer eye corners)
        left_face = landmarks[LandmarkIndices.LEFT_EYE_OUTER]
        right_face = landmarks[LandmarkIndices.RIGHT_EYE_OUTER]
        
        face_width = np.sqrt((left_face.x - right_face.x)**2 + (left_face.y - right_face.y)**2)
        if face_width == 0:
            return 0.0
        
        # Mouth width
        mouth_width = np.sqrt((left_corner.x - right_corner.x)**2 + (left_corner.y - right_corner.y)**2)
        
        # Smile ratio: mouth width relative to face width
        # Smiling typically increases this ratio
        smile_ratio = mouth_width / face_width
        
        # Normalize to 0-1 range (typical range is 0.4-0.7)
        normalized = (smile_ratio - 0.4) / 0.3
        return max(0, min(1, normalized))
    
    def _calculate_cheek_raise(self, landmarks) -> float:
        """
        Calculate cheek raise (Duchenne smile indicator).
        
        A genuine smile involves raising of the cheeks, causing eye area compression.
        This measures the vertical distance between cheek and lower eye.
        
        Higher value = more cheek raise (genuine smile indicator)
        """
        # Cheek landmarks (approximate using face mesh points near cheekbone)
        # Using points around the lower eye area
        left_cheek_idx = 50   # Near left cheek
        right_cheek_idx = 280  # Near right cheek
        
        left_cheek = landmarks[left_cheek_idx]
        right_cheek = landmarks[right_cheek_idx]
        
        # Lower eye landmarks
        left_lower_eye = landmarks[LandmarkIndices.LEFT_EYE_BOTTOM]
        right_lower_eye = landmarks[LandmarkIndices.RIGHT_EYE_BOTTOM]
        
        # Calculate vertical distance (y increases downward in image coords)
        # When cheeks are raised, cheek moves up (y decreases), closer to eye
        left_distance = left_cheek.y - left_lower_eye.y
        right_distance = right_cheek.y - right_lower_eye.y
        
        avg_distance = (left_distance + right_distance) / 2
        
        # Normalize: smaller distance = higher cheek raise
        # Typical range is 0.03-0.08 in normalized coords
        # Invert so higher = more raise
        normalized = 1.0 - (avg_distance - 0.02) / 0.06
        return max(0, min(1, normalized))
    
    def draw_landmarks(self, frame: np.ndarray, cues: Optional[AffectiveCues] = None) -> np.ndarray:
        """
        Draw face mesh landmarks on frame for visualization.
        
        Args:
            frame: Input BGR frame
            cues: Optional cues to display as text overlay
            
        Returns:
            Annotated frame
        """
        if not self.enabled:
            return frame
        
        output = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw eye contours
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        # Add cue overlay if provided
        if cues:
            self._draw_cue_overlay(output, cues)
        
        return output
    
    def _draw_cue_overlay(self, frame: np.ndarray, cues: AffectiveCues) -> None:
        """Draw cue values as text overlay."""
        y_offset = 30
        line_height = 25
        
        texts = [
            f"Face: {'Yes' if cues.face_detected else 'No'} ({cues.face_confidence:.2f})",
            f"Pitch: {cues.head_pitch:+.2f}  Yaw: {cues.head_yaw:+.2f}",
            f"Eye AR: {cues.avg_eye_aspect_ratio:.2f}  Gaze: {cues.gaze_ratio:.2f}",
            f"Mouth: {cues.mouth_aspect_ratio:.2f}  Brow: {cues.brow_height:.2f}",
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                frame, text, (10, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
    
    def release(self) -> None:
        """Release resources."""
        if self.enabled and hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            print("[CVProcessor] Released")
