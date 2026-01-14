"""
SeenZone Agent - Perceive-Reason-Act Loop
==========================================
Implements the core agent architecture using the classical
Perceive-Reason-Act (P-R-A) agent loop.

The agent cycle:
1. PERCEIVE: Gather percepts from all sensors (visual + text)
2. REASON: Apply FOL rules to determine state transitions
3. ACT: Generate empathetic response via LLM actuator

This module ties together the PEAS components, state-space model,
inference engine, and LLM actuator into a functioning agent loop.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .peas import (
    PEASAgent, Sensor, Actuator, SensorType, ActuatorType,
    EnvironmentState, PerformanceMeasure
)
from .state_space import StateSpace, EmotionalState
from .sensors.cv_processor import CVProcessor, AffectiveCues
from .sensors.cue_interpreter import CueInterpreter, SymbolicPredicates
from .reasoning.inference import InferenceEngine
from .actuators.responder import LLMResponder


@dataclass
class AgentConfig:
    """Configuration for the SeenZone agent."""
    cycle_delay_seconds: float = 0.1    # Delay between cycles
    max_cycles: int = 0                  # 0 = unlimited
    verbose: bool = True                 # Print debug info
    enable_cv: bool = True               # Enable CV processing
    enable_inference: bool = True        # Enable rule-based inference
    enable_llm: bool = True              # Enable LLM response generation


class SeenZoneAgent:
    """
    Main agent class implementing the Perceive-Reason-Act loop.
    
    Architecture:
    - Uses PEAS model for component organization
    - Maintains a state-space for emotional state tracking
    - Uses forward-chaining inference for reasoning
    - Uses LLM for empathetic response generation
    - Runs continuous P-R-A cycles until stopped
    
    Agent Theory Mapping:
    - Agent Function: Maps percept sequences to actions
    - Agent Program: This class implementation
    - Rationality: Maximize transitions toward S_goal
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the SeenZone agent."""
        self.config = config or AgentConfig()
        self.peas = PEASAgent(name="SeenZone")
        self.state_space = StateSpace()
        self.running = False
        self.cycle_count = 0
        
        # Store current cues for act phase
        self._current_cues: List[str] = []
        
        # CV processing components (Sprint 2)
        self.cv_processor: Optional[CVProcessor] = None
        self.cue_interpreter: Optional[CueInterpreter] = None
        
        # Inference engine (Sprint 3)
        self.inference_engine: Optional[InferenceEngine] = None
        
        # LLM responder (Sprint 4)
        self.llm_responder: Optional[LLMResponder] = None
        
        if self.config.enable_cv:
            self._init_cv_components()
        
        if self.config.enable_inference:
            self._init_inference_engine()
        
        if self.config.enable_llm:
            self._init_llm_responder()
        
        print(f"[Agent] SeenZone Agent initialized")
        print(f"[Agent] Initial state: {self.state_space.current_state}")
    
    def _init_cv_components(self) -> None:
        """Initialize CV processing components."""
        try:
            self.cv_processor = CVProcessor()
            self.cue_interpreter = CueInterpreter()
            print(f"[Agent] CV processing enabled")
        except Exception as e:
            print(f"[Agent] CV processing disabled: {e}")
            self.cv_processor = None
            self.cue_interpreter = None
    
    def _init_inference_engine(self) -> None:
        """Initialize the inference engine with default rules."""
        try:
            self.inference_engine = InferenceEngine(
                state_space=self.state_space,
                load_default_rules=True
            )
            print(f"[Agent] Inference engine enabled")
        except Exception as e:
            print(f"[Agent] Inference engine disabled: {e}")
            self.inference_engine = None
    
    def _init_llm_responder(self) -> None:
        """Initialize the LLM responder for response generation."""
        try:
            self.llm_responder = LLMResponder()
            llm_status = "LLM" if self.llm_responder.is_llm_available() else "fallback"
            print(f"[Agent] Response generation: {llm_status} mode")
        except Exception as e:
            print(f"[Agent] LLM responder disabled: {e}")
            self.llm_responder = None
    
    # =========================================================================
    # SENSOR/ACTUATOR REGISTRATION
    # =========================================================================
    
    def register_sensor(self, sensor: Sensor) -> None:
        """Register a sensor with the agent."""
        self.peas.add_sensor(sensor)
    
    def register_actuator(self, actuator: Actuator) -> None:
        """Register an actuator with the agent."""
        self.peas.add_actuator(actuator)
    
    # =========================================================================
    # PERCEIVE PHASE
    # =========================================================================
    
    def perceive(self) -> EnvironmentState:
        """
        Gather percepts from all registered sensors.
        
        This is the first phase of the P-R-A loop.
        Collects data from visual (webcam) and textual (user input) sensors.
        Extracts affective cues via CV processing.
        
        Returns:
            EnvironmentState containing all current percepts
        """
        env_state = EnvironmentState(timestamp=time.time())
        
        # Collect from visual sensor (webcam)
        visual_sensor = self.peas.get_sensor(SensorType.VISUAL)
        if visual_sensor and visual_sensor.is_available():
            env_state.visual_frame = visual_sensor.read()
            
            # Extract affective cues via CV processing
            if env_state.visual_frame is not None and self.cv_processor:
                cues = self.cv_processor.process(env_state.visual_frame)
                
                if self.cue_interpreter:
                    # NEW: Get baseline delta for laptop-aware detection
                    baseline_delta = None
                    if self.cv_processor.is_calibrated():
                        baseline_delta = self.cv_processor.get_baseline_delta(cues)
                    
                    # Pass baseline delta to interpreter for relative predicates
                    predicates = self.cue_interpreter.interpret(cues, baseline_delta)
                    env_state.visual_cues = {
                        "raw": cues.to_dict(),
                        "predicates": predicates.to_dict(),
                        "fol": predicates.to_fol_predicates(),
                        "summary": self.cue_interpreter.get_summary(predicates),
                        "sadness_evidence": predicates.sadness_evidence_count  # NEW: For logging
                    }
        
        # Collect from text sensor (if available)
        text_sensor = self.peas.get_sensor(SensorType.TEXTUAL)
        if text_sensor and text_sensor.is_available():
            env_state.text_input = text_sensor.read()
        
        if self.config.verbose:
            has_visual = "✓" if env_state.has_visual() else "✗"
            has_text = "✓" if env_state.has_text() else "✗"
            has_cues = "✓" if env_state.visual_cues else "✗"
            calibrated = "✓" if self.cv_processor and self.cv_processor.is_calibrated() else "⏳"
            print(f"[Perceive] Visual:{has_visual} Text:{has_text} Cues:{has_cues} Baseline:{calibrated}")
            
            # Print cue summary if available
            if env_state.visual_cues:
                print(f"[Perceive] {env_state.visual_cues.get('summary', 'No summary')}")
        
        return env_state
    
    # =========================================================================
    # REASON PHASE
    # =========================================================================
    
    def reason(self, env_state: EnvironmentState) -> Optional[EmotionalState]:
        """
        Apply forward-chaining inference to determine state transitions.
        
        This is the second phase of the P-R-A loop.
        Uses the inference engine with FOL-style rules.
        
        Args:
            env_state: Current environment percepts
            
        Returns:
            New state to transition to, or None if no transition
        """
        current = self.state_space.current_state
        
        if self.config.verbose:
            print(f"[Reason] Current state: {current}")
        
        # Get FOL predicates from CV processing
        fol_predicates = []
        if env_state.visual_cues and "fol" in env_state.visual_cues:
            fol_predicates = env_state.visual_cues["fol"]
        
        # Store for act phase (LLM needs these)
        self._current_cues = fol_predicates
        
        if self.config.verbose:
            if fol_predicates:
                print(f"[Reason] KB Facts: {', '.join(fol_predicates)}")
            else:
                print(f"[Reason] KB Facts: (none)")
        
        # Run inference if engine is available
        if self.inference_engine and fol_predicates:
            # Update knowledge base
            self.inference_engine.update_facts(fol_predicates)
            
            # Run inference
            new_state = self.inference_engine.infer()
            
            if self.config.verbose:
                # Show inference explanation
                result = self.inference_engine._last_result
                if result:
                    if result.fired_rule:
                        print(f"[Reason] Rule fired: {result.fired_rule.name} (priority={result.fired_rule.priority})")
                    if result.matching_rules:
                        print(f"[Reason] Matching rules: {len(result.matching_rules)}")
                    print(f"[Reason] Result: {result.explanation}")
            
            return new_state
        
        if self.config.verbose:
            print(f"[Reason] Inference: skipped (no predicates or engine)")
        
        return None
    
    # =========================================================================
    # ACT PHASE
    # =========================================================================
    
    def act(self, new_state: Optional[EmotionalState]) -> Optional[str]:
        """
        Generate and execute response based on reasoning outcome.
        
        This is the third phase of the P-R-A loop.
        Uses LLM responder to produce empathetic responses.
        Falls back to placeholder responses if LLM unavailable.
        
        Args:
            new_state: State to transition to (if any)
            
        Returns:
            Generated response text
        """
        # Execute state transition if needed
        if new_state is not None:
            old_state = self.state_space.current_state
            success = self.state_space.transition_to(new_state)
            if success:
                toward_goal = new_state.severity_level < old_state.severity_level
                self.peas.performance.record_transition(toward_goal)
                
                if self.config.verbose:
                    direction = "→ goal" if toward_goal else "← away"
                    print(f"[Act] Transition: {old_state.name} → {new_state.name} ({direction})")
        
        current = self.state_space.current_state
        
        # Generate response using LLM responder
        if self.llm_responder:
            response = self.llm_responder.generate_response(current, self._current_cues)
            
            if self.config.verbose:
                llm_mode = "LLM" if self.llm_responder.is_llm_available() else "fallback"
                print(f"[Act] State: {current} ({llm_mode})")
                print(f"[Act] Response: {response}")
        else:
            # Minimal fallback if responder not initialized
            response = "I'm here with you."
            
            if self.config.verbose:
                print(f"[Act] State: {current}")
                print(f"[Act] Response: {response} (no responder)")
        
        self.peas.performance.record_response()
        return response
    
    # =========================================================================
    # MAIN AGENT LOOP
    # =========================================================================
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete Perceive-Reason-Act cycle.
        
        Returns:
            Dict containing cycle results
        """
        self.cycle_count += 1
        self.peas.performance.record_cycle()
        
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"[Cycle {self.cycle_count}]")
        
        # Phase 1: PERCEIVE
        env_state = self.perceive()
        
        # Phase 2: REASON
        new_state = self.reason(env_state)
        
        # Phase 3: ACT
        response = self.act(new_state)
        
        return {
            "cycle": self.cycle_count,
            "environment": env_state,
            "new_state": new_state,
            "response": response,
            "state_summary": self.state_space.get_state_summary()
        }
    
    def run(self, cycles: Optional[int] = None) -> None:
        """
        Run the agent loop.
        
        Args:
            cycles: Number of cycles to run (None = use config)
        """
        max_cycles = cycles or self.config.max_cycles
        self.running = True
        
        print(f"\n[Agent] Starting agent loop (max_cycles={max_cycles or 'unlimited'})")
        print(f"[Agent] Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                result = self.run_cycle()
                
                # Check cycle limit
                if max_cycles > 0 and self.cycle_count >= max_cycles:
                    print(f"\n[Agent] Reached max cycles ({max_cycles})")
                    break
                
                # Delay between cycles
                time.sleep(self.config.cycle_delay_seconds)
                
        except KeyboardInterrupt:
            print(f"\n[Agent] Interrupted by user")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the agent and release resources."""
        self.running = False
        self.peas.release_all()
        
        # Release CV processor
        if self.cv_processor:
            self.cv_processor.release()
        
        # Print performance summary
        perf = self.peas.performance
        print(f"\n{'='*50}")
        print(f"[Agent] Session Summary")
        print(f"  Total cycles: {perf.total_cycles}")
        print(f"  State transitions: {perf.state_transitions}")
        print(f"  Goal approaches: {perf.goal_approaches}")
        print(f"  Responses generated: {perf.responses_generated}")
        print(f"  Stabilization ratio: {perf.stabilization_ratio:.2%}")
        
        # LLM stats if available
        if self.llm_responder:
            stats = self.llm_responder.get_stats()
            print(f"  LLM generations: {stats['llm_generations']}")
            print(f"  Fallback responses: {stats['fallback_responses']}")
        
        print(f"{'='*50}")
    
    # =========================================================================
    # CV VISUALIZATION
    # =========================================================================
    
    def get_annotated_frame(self, frame, cues: Optional[AffectiveCues] = None):
        """
        Get frame with CV annotations overlaid.
        
        Args:
            frame: Original webcam frame
            cues: Affective cues to display
            
        Returns:
            Annotated frame for visualization
        """
        if self.cv_processor and frame is not None:
            return self.cv_processor.draw_landmarks(frame, cues)
        return frame
    
    # =========================================================================
    # INFERENCE EXPLANATION
    # =========================================================================
    
    def get_inference_explanation(self) -> str:
        """Get detailed explanation of last inference cycle."""
        if self.inference_engine:
            return self.inference_engine.get_explanation()
        return "Inference engine not initialized"
