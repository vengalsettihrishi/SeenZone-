"""
SeenZone GUI - Unified Visualization
=====================================
OpenCV-based real-time display showing:
- Webcam feed with face mesh
- Current emotional state (visual anchor)
- Active symbolic cues
- Fired rule with priority
- LLM/fallback response
- Cycle counter

This is a presentation layer, not a UI framework.
All agent logic remains in agent.py.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .state_space import EmotionalState


# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

@dataclass
class DisplayConfig:
    """Configuration for the GUI display."""
    window_name: str = "SeenZone - Empathic Companion"
    width: int = 800
    height: int = 600
    
    # Colors (BGR format)
    bg_color: Tuple[int, int, int] = (30, 30, 30)       # Dark background
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White text
    state_color: Tuple[int, int, int] = (0, 255, 200)   # Cyan for state
    cue_color: Tuple[int, int, int] = (200, 200, 200)   # Light gray
    rule_color: Tuple[int, int, int] = (100, 255, 100)  # Green
    response_color: Tuple[int, int, int] = (255, 200, 100)  # Orange
    divider_color: Tuple[int, int, int] = (80, 80, 80)  # Gray divider
    
    # Font settings
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    state_font_scale: float = 1.0       # Larger for state
    normal_font_scale: float = 0.6      # Normal text
    small_font_scale: float = 0.5       # Small text
    font_thickness: int = 2


# State-to-color mapping for visual feedback
STATE_COLORS = {
    EmotionalState.S0_NEUTRAL: (200, 200, 200),        # Gray
    EmotionalState.S1_SADNESS_DETECTED: (255, 150, 100),    # Blue-ish
    EmotionalState.S2_DEPRESSION_SUSPECTED: (255, 100, 100), # Deeper blue
    EmotionalState.S3_HIGH_STRESS: (100, 100, 255),    # Red
    EmotionalState.S4_PROLONGED_STRESS: (80, 80, 255), # Darker red
    EmotionalState.S5_POSITIVE_STATE: (100, 255, 100), # Green
    EmotionalState.S6_DISTRESSED_SILENT: (200, 150, 255), # Purple
    EmotionalState.S_GOAL_STABILIZED: (200, 255, 200), # Light green
}


# =============================================================================
# GUI CLASS
# =============================================================================

class SeenZoneGUI:
    """
    OpenCV-based GUI for SeenZone visualization.
    
    Renders:
    - Webcam frame with face mesh overlay
    - State panel (visual anchor)
    - Cues panel
    - Rule fired panel  
    - Response panel
    - Cycle counter
    """
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        """Initialize the GUI."""
        self.config = config or DisplayConfig()
        self._running = False
        
        print(f"[GUI] Initialized: {self.config.window_name}")
    
    def render(self,
               frame: np.ndarray,
               state: EmotionalState,
               cues: List[str],
               rule_fired: Optional[str],
               response: str,
               cycle: int) -> np.ndarray:
        """
        Render a complete frame with all overlays.
        
        Args:
            frame: Webcam frame (will be resized)
            state: Current emotional state
            cues: List of active FOL predicates
            rule_fired: Name of fired rule (or None)
            response: LLM/fallback response text
            cycle: Current cycle number
            
        Returns:
            Rendered frame ready for display
        """
        cfg = self.config
        
        # Create output canvas
        canvas = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        canvas[:] = cfg.bg_color
        
        # Layout: video on top, info panel below
        video_height = int(cfg.height * 0.6)
        panel_height = cfg.height - video_height
        
        # Resize and place video frame
        if frame is not None:
            frame_resized = cv2.resize(frame, (cfg.width, video_height))
            canvas[0:video_height, 0:cfg.width] = frame_resized
        
        # Draw divider line
        cv2.line(canvas, (0, video_height), (cfg.width, video_height), 
                 cfg.divider_color, 2)
        
        # Draw info panel
        self._draw_info_panel(canvas, video_height, state, cues, 
                              rule_fired, response, cycle)
        
        # Draw title bar on video
        self._draw_title_bar(canvas)
        
        return canvas
    
    def _draw_title_bar(self, canvas: np.ndarray) -> None:
        """Draw title bar at top of video."""
        cfg = self.config
        
        # Semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (cfg.width, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        # Title text
        cv2.putText(canvas, "SeenZone - Empathic Companion", (10, 25),
                    cfg.font, cfg.normal_font_scale, cfg.text_color, 1)
    
    def _draw_info_panel(self,
                         canvas: np.ndarray,
                         y_start: int,
                         state: EmotionalState,
                         cues: List[str],
                         rule_fired: Optional[str],
                         response: str,
                         cycle: int) -> None:
        """Draw the information panel below the video."""
        cfg = self.config
        
        # Panel background (slightly lighter)
        panel_bg = tuple(min(255, c + 20) for c in cfg.bg_color)
        cv2.rectangle(canvas, (0, y_start), (cfg.width, cfg.height), panel_bg, -1)
        
        y = y_start + 30
        left_margin = 15
        
        # === STATE (Visual Anchor - Prominent) ===
        state_color = STATE_COLORS.get(state, cfg.state_color)
        state_text = f"STATE: {state.name}"
        cv2.putText(canvas, state_text, (left_margin, y),
                    cfg.font, cfg.state_font_scale, state_color, cfg.font_thickness)
        y += 35
        
        # === CUES ===
        cue_str = ", ".join(cues[:5]) if cues else "(none detected)"  # Limit to 5
        cv2.putText(canvas, f"Cues: {cue_str}", (left_margin, y),
                    cfg.font, cfg.normal_font_scale, cfg.cue_color, 1)
        y += 25
        
        # === RULE FIRED ===
        if rule_fired:
            rule_text = f"Rule Fired: {rule_fired}"
        else:
            rule_text = "Rule Fired: (none)"
        cv2.putText(canvas, rule_text, (left_margin, y),
                    cfg.font, cfg.normal_font_scale, cfg.rule_color, 1)
        y += 25
        
        # === CYCLE COUNTER (top right of panel) ===
        cycle_text = f"Cycle: {cycle}"
        (text_w, _), _ = cv2.getTextSize(cycle_text, cfg.font, cfg.small_font_scale, 1)
        cv2.putText(canvas, cycle_text, (cfg.width - text_w - 15, y_start + 25),
                    cfg.font, cfg.small_font_scale, cfg.cue_color, 1)
        
        # === DIVIDER ===
        cv2.line(canvas, (left_margin, y), (cfg.width - left_margin, y), 
                 cfg.divider_color, 1)
        y += 20
        
        # === RESPONSE ===
        cv2.putText(canvas, "Response:", (left_margin, y),
                    cfg.font, cfg.small_font_scale, cfg.response_color, 1)
        y += 22
        
        # Wrap response text if too long
        wrapped = self._wrap_text(response, cfg.width - 30, cfg.font, cfg.normal_font_scale)
        for line in wrapped[:2]:  # Max 2 lines
            cv2.putText(canvas, line, (left_margin, y),
                        cfg.font, cfg.normal_font_scale, cfg.text_color, 1)
            y += 22
    
    def _wrap_text(self, text: str, max_width: int, 
                   font: int, scale: float) -> List[str]:
        """Wrap text to fit within max_width pixels."""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            (w, _), _ = cv2.getTextSize(test_line, font, scale, 1)
            
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def show(self, frame: np.ndarray) -> bool:
        """
        Display a frame and check for quit key.
        
        Args:
            frame: Rendered frame to display
            
        Returns:
            True if should continue, False if user pressed 'q'
        """
        cv2.imshow(self.config.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def close(self) -> None:
        """Close the display window."""
        cv2.destroyAllWindows()
        print("[GUI] Closed")


# =============================================================================
# INTEGRATED RUN FUNCTION
# =============================================================================

def run_unified_demo(agent, webcam, max_cycles: int = 0):
    """
    Run the full agent with unified GUI visualization.
    
    Args:
        agent: SeenZoneAgent instance
        webcam: WebcamSensor instance
        max_cycles: Maximum cycles (0 = unlimited)
    """
    gui = SeenZoneGUI()
    
    print("\n[Demo] Starting unified visualization")
    print("[Demo] Press 'q' to quit\n")
    
    cycle = 0
    
    try:
        while True:
            cycle += 1
            
            # Get frame
            frame = webcam.read()
            if frame is None:
                print("[Demo] Failed to read frame")
                break
            
            # Process with CV if available
            cues = []
            if agent.cv_processor and agent.cue_interpreter:
                raw_cues = agent.cv_processor.process(frame)
                predicates = agent.cue_interpreter.interpret(raw_cues)
                cues = predicates.to_fol_predicates()
                
                # Draw face mesh on frame
                frame = agent.cv_processor.draw_landmarks(frame)
            
            # Run reasoning
            if agent.inference_engine and cues:
                agent.inference_engine.update_facts(cues)
                new_state = agent.inference_engine.infer()
                
                # Get fired rule name
                rule_fired = None
                if agent.inference_engine._last_result and agent.inference_engine._last_result.fired_rule:
                    rule = agent.inference_engine._last_result.fired_rule
                    rule_fired = f"{rule.name} (priority={rule.priority})"
                
                # Execute state transition
                if new_state:
                    agent.state_space.transition_to(new_state)
            else:
                rule_fired = None
            
            # Get current state
            current_state = agent.state_space.current_state
            
            # Generate response
            if agent.llm_responder:
                response = agent.llm_responder.generate_response(current_state, cues)
            else:
                response = "I'm here with you."
            
            # Render GUI
            rendered = gui.render(
                frame=frame,
                state=current_state,
                cues=[c.split("(")[0] for c in cues],  # Clean cue names
                rule_fired=rule_fired,
                response=response,
                cycle=cycle
            )
            
            # Show and check for quit
            if not gui.show(rendered):
                break
            
            # Check cycle limit
            if max_cycles > 0 and cycle >= max_cycles:
                print(f"[Demo] Reached max cycles ({max_cycles})")
                break
                
    except KeyboardInterrupt:
        print("\n[Demo] Interrupted")
    finally:
        gui.close()
        print(f"[Demo] Completed {cycle} cycles")
