"""
Webcam Sensor Module
====================
Implements the visual sensor using OpenCV for webcam capture.

PEAS Mapping: SENSOR component
- Captures raw video frames from the user's webcam
- Provides frames to the CV processing pipeline (Sprint 2)
- Respects privacy: frames stay local, never sent to LLM

This sensor provides the visual input for affective cue extraction.
The actual CV processing (face detection, pose estimation) is
implemented in Sprint 2.
"""

import cv2
import numpy as np
from typing import Optional, Any

from ..peas import Sensor, SensorType


class WebcamSensor(Sensor):
    """
    Webcam sensor for capturing visual percepts.
    
    Uses OpenCV VideoCapture to access the system's default webcam.
    Returns raw frames as numpy arrays that can be processed by
    the CV module in Sprint 2.
    
    Privacy Note:
    - All processing is LOCAL
    - Frames are never transmitted to external APIs
    - Only symbolic cues (not images) are used for LLM prompting
    """
    
    def __init__(self, camera_index: int = 0, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize the webcam sensor.
        
        Args:
            camera_index: Camera device index (0 = default webcam)
            frame_width: Desired frame width
            frame_height: Desired frame height
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture: Optional[cv2.VideoCapture] = None
        self._available = False
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the video capture device."""
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if self.capture.isOpened():
                # Set frame dimensions
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                
                # Test read to confirm camera works
                ret, _ = self.capture.read()
                self._available = ret
                
                if self._available:
                    actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[WebcamSensor] Initialized: {actual_width}x{actual_height}")
                else:
                    print(f"[WebcamSensor] Warning: Camera opened but read failed")
            else:
                print(f"[WebcamSensor] Warning: Could not open camera {self.camera_index}")
                self._available = False
                
        except Exception as e:
            print(f"[WebcamSensor] Error: {e}")
            self._available = False
    
    @property
    def sensor_type(self) -> SensorType:
        """Returns the sensor type (VISUAL)."""
        return SensorType.VISUAL
    
    def is_available(self) -> bool:
        """Check if the webcam is operational."""
        return self._available and self.capture is not None and self.capture.isOpened()
    
    def read(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the webcam.
        
        Returns:
            numpy array (BGR format) if successful, None otherwise
        """
        if not self.is_available():
            return None
        
        ret, frame = self.capture.read()
        
        if ret and frame is not None:
            return frame
        else:
            return None
    
    def read_rgb(self) -> Optional[np.ndarray]:
        """
        Capture a frame and convert to RGB format.
        
        Returns:
            numpy array (RGB format) if successful, None otherwise
        """
        frame = self.read()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def release(self) -> None:
        """Release the webcam resource."""
        if self.capture is not None:
            self.capture.release()
            self._available = False
            print(f"[WebcamSensor] Released")
    
    def get_frame_info(self) -> dict:
        """Get information about the current capture settings."""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "camera_index": self.camera_index,
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.capture.get(cv2.CAP_PROP_FPS)
        }


def display_webcam_feed(sensor: WebcamSensor, window_name: str = "SeenZone - Webcam Feed") -> None:
    """
    Display live webcam feed in a window.
    
    Utility function for testing and verification.
    Press 'q' to quit.
    
    Args:
        sensor: WebcamSensor instance
        window_name: Name for the display window
    """
    if not sensor.is_available():
        print("[Display] Webcam not available")
        return
    
    print(f"[Display] Showing webcam feed. Press 'q' to quit.")
    
    while True:
        frame = sensor.read()
        
        if frame is None:
            print("[Display] Failed to read frame")
            break
        
        # Add overlay text showing this is SeenZone
        cv2.putText(
            frame, 
            "SeenZone - Visual Sensor Active", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
