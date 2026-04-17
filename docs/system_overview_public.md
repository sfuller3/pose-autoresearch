# System Overview — Event Detection for Elder Care

## Privacy-First Design

We convert video into simple stick figure representations in real-time. The stick figure captures body position and movement — but the original image is never stored or analyzed by the event detection model. This means the system works without recording identifiable video of residents.

## What the System Sees

The system tracks two things:

1. **The person** — represented as 17 body points (head, shoulders, elbows, wrists, hips, knees, ankles) forming a stick figure, updated 30 times per second
2. **The room** — furniture and fixtures like beds, chairs, tables, doors, wheelchairs, and walkers, detected as simple labeled rectangles

## How It Detects Events

The system watches the stick figure over a sliding window of several seconds. It uses multiple detection channels tuned to different speeds — fast channels for sudden events like falls (under a second), medium channels for activities like eating (a few seconds), and slow channels for patterns like wandering (many seconds).

The room context informs the detection. A person lowering onto a bed is different from a person collapsing onto the floor, even though the stick figure motion may look similar. By understanding what objects are nearby, the system reduces false alarms.

The system combines machine learning — trained on thousands of real movement sequences — with expert-defined safety rules. The ML learns subtle patterns humans might miss. The rules encode clinical knowledge that ensures reliable behavior from day one.

## What It Detects

Seven event types relevant to elder care: **falls**, **eating**, **unstable gait**, **wandering**, **aggression**, **cooperative activity**, and **postural transitions** (sitting/standing).

## On-Premise Processing

All processing happens on a small device within the facility — no cloud required. This keeps latency under 50 milliseconds, keeps data on-premise, and operates without an internet connection.
