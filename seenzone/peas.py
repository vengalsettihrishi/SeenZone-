"""
PEAS Framework Definitions
==========================
Implements the classical PEAS (Performance, Environment, Actuators, Sensors)
model for agent theory.

PEAS Mapping for SeenZone:
- Performance: Emotional stabilization metrics (transitions toward S_goal)
- Environment: User's facial/postural presentation + text input
- Actuators: Text-based empathetic responses
- Sensors: Webcam (visual cues) + text input (verbal cues)

This module provides the abstract interfaces and composite agent class
that binds all PEAS components together.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# =============================================================================
# SENSOR INTERFACE
# =============================================================================
# Maps to: PEAS "Sensors" component
# Purpose: Provides percepts from the environment to the agent

class SensorType(Enum):
    """Types of sensors available to the agent."""
    VISUAL = "visual"      # Webcam input
    TEXTUAL = "textual"    # User text input


class Sensor(ABC):
    """
    Abstract base class for all sensors.
    
    In agent theory, sensors gather percepts from the environment.
    Each sensor implementation captures a specific modality of input.
    """
    
    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """Returns the type of this sensor."""
        pass
    
    @abstractmethod
    def read(self) -> Optional[Any]:
        """
        Capture current percept from the environment.
        
        Returns:
            Percept data (format depends on sensor type), or None if unavailable
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the sensor is operational."""
        pass
    
    def release(self) -> None:
        """Release sensor resources. Override if cleanup needed."""
        pass


# =============================================================================
# ACTUATOR INTERFACE
# =============================================================================
# Maps to: PEAS "Actuators" component
# Purpose: Executes actions in the environment based on agent decisions

class ActuatorType(Enum):
    """Types of actuators available to the agent."""
    TEXT_RESPONSE = "text_response"  # Generate text output


class Actuator(ABC):
    """
    Abstract base class for all actuators.
    
    In agent theory, actuators perform actions that affect the environment.
    For SeenZone, the primary actuator generates empathetic text responses.
    """
    
    @property
    @abstractmethod
    def actuator_type(self) -> ActuatorType:
        """Returns the type of this actuator."""
        pass
    
    @abstractmethod
    def execute(self, action: Dict[str, Any]) -> Any:
        """
        Execute an action in the environment.
        
        Args:
            action: Action specification (format depends on actuator type)
            
        Returns:
            Result of the action execution
        """
        pass


# =============================================================================
# ENVIRONMENT MODEL
# =============================================================================
# Maps to: PEAS "Environment" component
# Purpose: Represents the current state of the environment as perceived

@dataclass
class EnvironmentState:
    """
    Represents the current environment state as perceived by the agent.
    
    This is a snapshot of all available percepts at a given moment.
    The agent uses this to make decisions about state transitions.
    
    Attributes:
        visual_frame: Raw frame from webcam (numpy array or None)
        visual_cues: Extracted symbolic cues from CV processing (Sprint 2+)
        text_input: Latest text input from user (if any)
        timestamp: When this state was captured
    """
    visual_frame: Optional[Any] = None          # Raw webcam frame
    visual_cues: Dict[str, Any] = field(default_factory=dict)  # Symbolic cues
    text_input: Optional[str] = None            # User's text message
    timestamp: float = 0.0                       # Capture time
    
    def has_visual(self) -> bool:
        """Check if visual data is available."""
        return self.visual_frame is not None
    
    def has_text(self) -> bool:
        """Check if text input is available."""
        return self.text_input is not None and len(self.text_input.strip()) > 0


# =============================================================================
# PERFORMANCE MEASURE
# =============================================================================
# Maps to: PEAS "Performance" component
# Purpose: Evaluates how well the agent is achieving its goals

@dataclass
class PerformanceMeasure:
    """
    Tracks agent performance metrics.
    
    In SeenZone, performance is measured by:
    1. Successful state transitions toward S_goal (emotional stabilization)
    2. Appropriate response generation
    3. User engagement metrics
    
    Note: This is a simplified academic model, not a clinical assessment.
    """
    total_cycles: int = 0                        # Total agent cycles run
    state_transitions: int = 0                   # Number of state changes
    goal_approaches: int = 0                     # Transitions toward S_goal
    goal_diversions: int = 0                     # Transitions away from S_goal
    responses_generated: int = 0                 # Total responses produced
    
    def record_transition(self, toward_goal: bool) -> None:
        """Record a state transition."""
        self.state_transitions += 1
        if toward_goal:
            self.goal_approaches += 1
        else:
            self.goal_diversions += 1
    
    def record_cycle(self) -> None:
        """Record completion of an agent cycle."""
        self.total_cycles += 1
    
    def record_response(self) -> None:
        """Record a generated response."""
        self.responses_generated += 1
    
    @property
    def stabilization_ratio(self) -> float:
        """
        Ratio of goal-approaching transitions.
        Higher is better (agent is helping user move toward stability).
        """
        if self.state_transitions == 0:
            return 0.0
        return self.goal_approaches / self.state_transitions


# =============================================================================
# PEAS AGENT COMPOSITE
# =============================================================================

@dataclass
class PEASAgent:
    """
    Composite class binding all PEAS components.
    
    This represents the complete agent configuration as defined by
    the PEAS model. It holds references to all sensors, actuators,
    and tracks performance metrics.
    
    Usage:
        agent = PEASAgent(name="SeenZone")
        agent.add_sensor(webcam_sensor)
        agent.add_actuator(text_responder)
    """
    name: str = "SeenZone"
    sensors: List[Sensor] = field(default_factory=list)
    actuators: List[Actuator] = field(default_factory=list)
    performance: PerformanceMeasure = field(default_factory=PerformanceMeasure)
    
    def add_sensor(self, sensor: Sensor) -> None:
        """Register a sensor with the agent."""
        self.sensors.append(sensor)
        print(f"[PEAS] Registered sensor: {sensor.sensor_type.value}")
    
    def add_actuator(self, actuator: Actuator) -> None:
        """Register an actuator with the agent."""
        self.actuators.append(actuator)
        print(f"[PEAS] Registered actuator: {actuator.actuator_type.value}")
    
    def get_sensor(self, sensor_type: SensorType) -> Optional[Sensor]:
        """Get a sensor by type."""
        for sensor in self.sensors:
            if sensor.sensor_type == sensor_type:
                return sensor
        return None
    
    def get_actuator(self, actuator_type: ActuatorType) -> Optional[Actuator]:
        """Get an actuator by type."""
        for actuator in self.actuators:
            if actuator.actuator_type == actuator_type:
                return actuator
        return None
    
    def release_all(self) -> None:
        """Release all sensor resources."""
        for sensor in self.sensors:
            sensor.release()
        print(f"[PEAS] All sensors released")
