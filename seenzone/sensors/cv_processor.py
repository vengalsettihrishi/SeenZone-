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
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2

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
    
    All processing is performed locally. No data is transmitted externally.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 refine_landmarks: bool = True):
        """
        Initialize the CV processor.
        
        Args:
            min_detection_confidence: Minimum face detection confidence
            min_tracking_confidence: Minimum tracking confidence
            refine_landmarks: Whether to refine eye/lip landmarks
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
        
        print(f"[CVProcessor] Initialized (detection={min_detection_confidence}, tracking={min_tracking_confidence})")
    
    def process(self, frame: np.ndarray) -> AffectiveCues:
        """
        Process a frame and extract affective cues.
        
        Args:
            frame: BGR image from OpenCV (numpy array)
            
        Returns:
            AffectiveCues containing all extracted measurements
        """
        import time
        
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
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
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
        
        # Extract cues
        cues.head_pitch, cues.head_yaw, cues.head_roll = self._estimate_head_pose(landmarks, w, h)
        cues.left_eye_aspect_ratio = self._calculate_eye_aspect_ratio(landmarks, LandmarkIndices.LEFT_EYE)
        cues.right_eye_aspect_ratio = self._calculate_eye_aspect_ratio(landmarks, LandmarkIndices.RIGHT_EYE)
        cues.gaze_ratio = self._calculate_gaze_ratio(landmarks, w)
        cues.mouth_aspect_ratio = self._calculate_mouth_aspect_ratio(landmarks)
        cues.brow_height = self._calculate_brow_height(landmarks)
        
        return cues
    
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
