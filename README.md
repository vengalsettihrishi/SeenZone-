# SeenZone - Dual-System Empathic Companion

An academic AI project demonstrating classical AI frameworks applied to mental wellness support.

## Overview

SeenZone is a **Proof of Concept** that combines:
- **Computer Vision** — extracts non-verbal affective cues from webcam
- **Knowledge Representation** — FOL-style rules for state inference
- **LLM Integration** — generates empathetic, context-aware responses

> ⚠️ **Disclaimer**: This is an academic project, NOT a clinical tool. It makes no diagnostic claims.

## Architecture

```
Webcam → CV cues → FOL rules → State transition → LLM prompt → Response
```

### PEAS Model
| Component | Implementation |
|-----------|----------------|
| **Performance** | State transitions toward emotional stabilization |
| **Environment** | User's visual presentation + text input |
| **Actuators** | LLM-generated empathetic responses |
| **Sensors** | Webcam (OpenCV + MediaPipe) |

### State Space
```
S0: Neutral → S1: Sadness → S2: Depression Suspected
            → S3: Stress  → S4: Prolonged Stress
            → S5: Positive → S_goal: Stabilized
            → S6: Distressed Silent
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key (Optional)
Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

> Without an API key, the agent runs in fallback mode with placeholder responses.

### 3. Run
```bash
python main.py
```

## Project Structure

```
seenzone/
├── peas.py              # PEAS framework interfaces
├── state_space.py       # 8-state emotional FSM
├── agent.py             # Perceive-Reason-Act loop
├── sensors/
│   ├── webcam.py        # OpenCV camera capture
│   ├── cv_processor.py  # MediaPipe face mesh
│   └── cue_interpreter.py  # Symbolic predicates
├── reasoning/
│   ├── knowledge_base.py   # Fact storage
│   ├── rules.py            # Rule engine
│   └── inference.py        # Forward chaining
└── actuators/
    ├── llm_client.py       # Groq API wrapper
    ├── prompt_builder.py   # State-conditioned prompts
    └── responder.py        # Response generation
```

## Academic Framework Mapping

| AI Concept | Implementation |
|------------|----------------|
| PEAS Model | `peas.py` |
| State-Space Search | `state_space.py` |
| Knowledge Representation | `reasoning/` (FOL-style rules) |
| Perceive-Reason-Act | `agent.py` |
| Forward Chaining | `inference.py` |

## Privacy

- All CV processing is **local** — no images sent externally
- Only symbolic predicates (text) are sent to LLM
- No data persistence across sessions

## License

Academic use only.
