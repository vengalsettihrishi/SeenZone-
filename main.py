"""
SeenZone - Main Entry Point
============================
Demonstrates the complete agent with all components integrated.

Usage:
    python main.py

Demo Modes:
1. CV Demo - Face mesh + cue visualization
2. Agent Loop - P-R-A cycles with CV overlay
3. Agent Loop - Terminal only
4. Unified Demo - Full agent with integrated GUI (recommended)
"""

import cv2
import sys

from seenzone.agent import SeenZoneAgent, AgentConfig
from seenzone.sensors.webcam import WebcamSensor
from seenzone.gui import SeenZoneGUI, run_unified_demo


def run_cv_demo(webcam, agent):
    """Run CV demo with live landmark visualization."""
    from seenzone.sensors.cv_processor import CVProcessor
    from seenzone.sensors.cue_interpreter import CueInterpreter
    
    cv_processor = agent.cv_processor or CVProcessor()
    cue_interpreter = agent.cue_interpreter or CueInterpreter()
    
    print("\n[Demo] Starting CV visualization demo...")
    print("[Demo] Press 'q' to quit\n")
    
    while True:
        frame = webcam.read()
        if frame is None:
            break
        
        cues = cv_processor.process(frame)
        predicates = cue_interpreter.interpret(cues)
        annotated = cv_processor.draw_landmarks(frame, cues)
        
        # Add predicate summary
        summary = cue_interpreter.get_summary(predicates)
        cv2.putText(annotated, summary, (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("SeenZone - CV Demo", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


def run_agent_terminal(agent):
    """Run agent in terminal-only mode."""
    print("\n[Demo] Starting agent loop (terminal mode)...")
    agent.run(cycles=10)


def main():
    """Main entry point for SeenZone demonstration."""
    print("=" * 60)
    print("  SeenZone - Dual-System Empathic Companion")
    print("  Final Integration Demo")
    print("=" * 60)
    print()
    
    # Initialize webcam
    print("[Main] Initializing webcam...")
    webcam = WebcamSensor(camera_index=0)
    if not webcam.is_available():
        print("[Main] ERROR: Webcam not available.")
        return
    print("[Main] Webcam ready")
    
    # Initialize agent
    print("[Main] Initializing agent...")
    config = AgentConfig(
        cycle_delay_seconds=0.5,
        max_cycles=0,  # Unlimited for demo
        verbose=False,  # Reduce noise for GUI mode
        enable_cv=True,
        enable_inference=True,
        enable_llm=True,
    )
    agent = SeenZoneAgent(config=config)
    agent.register_sensor(webcam)
    
    # Show menu
    print("\n" + "=" * 60)
    print("Select demo mode:")
    print("  1. CV Demo (face mesh + cues)")
    print("  2. Agent Loop (terminal + video)")
    print("  3. Agent Loop (terminal only)")
    print("  4. Unified Demo (recommended)")
    print("  q. Quit")
    print("=" * 60)
    
    choice = input("Enter choice (1/2/3/4/q): ").strip().lower()
    
    if choice == 'q':
        print("[Main] Exiting.")
        webcam.release()
        return
    
    if choice == '1':
        run_cv_demo(webcam, agent)
    
    elif choice == '2':
        # Agent with video overlay (old mode)
        agent.config.verbose = True
        print("\n[Demo] Starting agent with CV visualization...")
        print("[Demo] Press 'q' to stop\n")
        
        try:
            while True:
                frame = webcam.read()
                result = agent.run_cycle()
                
                if frame is not None and agent.cv_processor:
                    cues = agent.cv_processor.process(frame)
                    annotated = agent.cv_processor.draw_landmarks(frame, cues)
                    cv2.imshow("SeenZone - Agent", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            agent.stop()
    
    elif choice == '3':
        agent.config.verbose = True
        run_agent_terminal(agent)
    
    elif choice == '4':
        # Unified demo (recommended)
        run_unified_demo(agent, webcam)
    
    else:
        print(f"[Main] Unknown choice: {choice}")
    
    # Cleanup
    print("\n[Main] Demo complete.")
    webcam.release()
    if agent.cv_processor:
        agent.cv_processor.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
        sys.exit(0)
