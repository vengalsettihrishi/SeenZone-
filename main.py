"""
SeenZone - Main Entry Point
============================
Demonstrates the agent initialization and CV-enabled run loop.

Usage:
    python main.py

Sprint 2 Features:
- CV processing with MediaPipe Face Mesh
- Affective cue extraction (head pose, eye AR, gaze, etc.)
- Symbolic predicate generation for FOL rules
- Visual debug overlay showing landmarks and cue values
"""

import cv2
import sys

from seenzone.agent import SeenZoneAgent, AgentConfig
from seenzone.sensors.webcam import WebcamSensor
from seenzone.sensors.cv_processor import CVProcessor
from seenzone.sensors.cue_interpreter import CueInterpreter
from seenzone.state_space import EmotionalState


def run_cv_demo(webcam: WebcamSensor, cv_processor: CVProcessor, cue_interpreter: CueInterpreter):
    """
    Run CV demo with live landmark visualization.
    
    Shows:
    - Face mesh overlay
    - Real-time affective cue values
    - Symbolic predicate generation
    
    Press 'q' to quit.
    """
    print("\n[Demo] Starting CV visualization demo...")
    print("[Demo] Press 'q' to quit\n")
    
    while True:
        frame = webcam.read()
        if frame is None:
            print("[Demo] Failed to read frame")
            break
        
        # Process frame for affective cues
        cues = cv_processor.process(frame)
        
        # Convert to symbolic predicates
        predicates = cue_interpreter.interpret(cues)
        
        # Draw landmarks and cue overlay
        annotated = cv_processor.draw_landmarks(frame, cues)
        
        # Add predicate summary at bottom
        summary = cue_interpreter.get_summary(predicates)
        cv2.putText(
            annotated, summary, (10, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        
        # Add FOL predicates
        fol = predicates.to_fol_predicates()
        if fol:
            fol_text = "FOL: " + ", ".join(fol[:3])  # Show first 3
            if len(fol) > 3:
                fol_text += f" (+{len(fol)-3} more)"
            cv2.putText(
                annotated, fol_text, (10, annotated.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
        
        # Show frame
        cv2.imshow("SeenZone - CV Demo", annotated)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("[Demo] CV demo ended")


def run_agent_with_cv(agent: SeenZoneAgent, webcam: WebcamSensor):
    """
    Run agent loop with CV-annotated video output.
    
    Shows live video with:
    - Face mesh overlay
    - Agent cycle information
    - State and cue information
    """
    print("\n[Demo] Starting agent with CV visualization...")
    print("[Demo] Press 'q' to stop\n")
    
    agent.running = True
    
    try:
        while agent.running:
            # Get frame for visualization
            frame = webcam.read()
            
            # Run agent cycle
            result = agent.run_cycle()
            
            # Annotate frame if available
            if frame is not None and agent.cv_processor:
                cues = agent.cv_processor.process(frame)
                annotated = agent.cv_processor.draw_landmarks(frame, cues)
                
                # Add agent state info
                state_text = f"State: {agent.state_space.current_state}"
                cv2.putText(
                    annotated, state_text, (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                cv2.imshow("SeenZone - Agent + CV", annotated)
            
            # Check for quit (non-blocking)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check cycle limit
            if agent.config.max_cycles > 0 and agent.cycle_count >= agent.config.max_cycles:
                print(f"\n[Agent] Reached max cycles")
                break
                
    except KeyboardInterrupt:
        print("\n[Agent] Interrupted")
    finally:
        cv2.destroyAllWindows()
        agent.stop()


def main():
    """Main entry point for SeenZone demonstration."""
    print("=" * 60)
    print("  SeenZone - Dual-System Empathic Companion")
    print("  Sprint 2: CV Module - Affective Cue Extraction")
    print("=" * 60)
    print()
    
    # =========================================================================
    # STEP 1: Initialize Components
    # =========================================================================
    print("[Main] Initializing components...")
    
    # Webcam
    webcam = WebcamSensor(camera_index=0)
    if not webcam.is_available():
        print("[Main] ERROR: Webcam not available. Cannot run CV demo.")
        return
    
    # CV Processor
    try:
        cv_processor = CVProcessor()
        cue_interpreter = CueInterpreter()
        cv_available = cv_processor.enabled
    except Exception as e:
        print(f"[Main] CV initialization failed: {e}")
        cv_available = False
    
    # Agent
    config = AgentConfig(
        cycle_delay_seconds=0.5,
        max_cycles=20,
        verbose=True,
        enable_cv=cv_available
    )
    agent = SeenZoneAgent(config=config)
    agent.register_sensor(webcam)
    
    # =========================================================================
    # STEP 2: Show Menu
    # =========================================================================
    print("\n" + "=" * 60)
    print("Select demo mode:")
    print("  1. CV Demo (face mesh + cue visualization)")
    print("  2. Agent Loop (P-R-A cycles with CV)")
    print("  3. Agent Loop (terminal only, no video)")
    print("  q. Quit")
    print("=" * 60)
    
    choice = input("Enter choice (1/2/3/q): ").strip().lower()
    
    if choice == 'q':
        print("[Main] Exiting.")
        webcam.release()
        return
    
    # =========================================================================
    # STEP 3: Run Selected Demo
    # =========================================================================
    
    if choice == '1':
        if cv_available:
            run_cv_demo(webcam, cv_processor, cue_interpreter)
        else:
            print("[Main] CV not available. Install mediapipe: pip install mediapipe")
    
    elif choice == '2':
        if cv_available:
            run_agent_with_cv(agent, webcam)
        else:
            print("[Main] CV not available for video overlay. Running terminal-only.")
            agent.run(cycles=10)
    
    elif choice == '3':
        agent.run(cycles=10)
    
    else:
        print(f"[Main] Unknown choice: {choice}")
    
    # =========================================================================
    # STEP 4: Cleanup
    # =========================================================================
    print("\n[Main] Demo complete.")
    webcam.release()
    if cv_available:
        cv_processor.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
        sys.exit(0)
