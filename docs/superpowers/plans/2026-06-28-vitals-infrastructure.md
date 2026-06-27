# Edge/Cloud Infrastructure Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make alert/event delivery resilient: a durable store-and-forward queue so the data record survives a cloud outage, a pluggable alert-sink architecture, a LAN life-safety alarm sink for critical events, and an edge self-health heartbeat — all preserving the rule that the edge never serves data to workers/families directly.

**Architecture:** A new `edge_sync.py` holds the queue, the sink interface and concrete sinks (`ConsoleSink`, `CloudSink`, `LANAlarmSink`), severity classification, and the heartbeat. `stream_detect.py`'s `AlertDispatcher` is refactored to delegate to a list of sinks (behavior-preserving for the default), and `run_pipeline`/`main` gain CLI wiring. This plan is independent of Plan 1 (the signal chain); they share only that critical vitals alerts, once Plan 1 emits them, will flow through these sinks.

**Tech Stack:** Python 3.9, standard library only (`json`, `urllib`, `collections`, `threading`, `time`). Tests via `.venv/bin/python -m pytest` with monkeypatched network calls — no real sockets.

**Conventions:**
- All commands from repo root `/Users/samfuller/Projects/pose-autoresearch`.
- `prepare.py` IMMUTABLE.
- Existing tests must keep passing after every task.
- No real network in tests — inject/monkeypatch the HTTP POST.
- Trust boundary is non-negotiable: sinks only PUSH outward (cloud) or alarm to on-prem facility infra (LAN); nothing here opens an inbound port on the edge.

**File structure:**
- Create `edge_sync.py` — queue, sinks, severity, heartbeat (Tasks 1-6).
- Modify `stream_detect.py` — `AlertDispatcher` → sink list; CLI wiring (Tasks 7-8).
- Modify `tests/test_pipeline.py` — append test classes per task.

---

### Task 1: Severity classification

**Files:**
- Create: `edge_sync.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
# ============================================================================
# EDGE / CLOUD INFRASTRUCTURE TESTS
# ============================================================================


class TestSeverity:
    def test_fall_is_critical(self):
        from edge_sync import classify_severity, Severity
        assert classify_severity({"class": "fall"}) == Severity.CRITICAL

    def test_unresponsive_is_critical(self):
        from edge_sync import classify_severity, Severity
        assert classify_severity({"event": "unresponsive"}) == Severity.CRITICAL

    def test_abnormal_vitals_is_critical(self):
        from edge_sync import classify_severity, Severity
        assert classify_severity({"event": "abnormal_vitals", "hr_bpm": 35}) == Severity.CRITICAL

    def test_routine_vitals_is_routine(self):
        from edge_sync import classify_severity, Severity
        assert classify_severity({"event": "vitals", "hr_bpm": 72}) == Severity.ROUTINE

    def test_unknown_is_routine(self):
        from edge_sync import classify_severity, Severity
        assert classify_severity({"class": "eating"}) == Severity.ROUTINE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestSeverity -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edge_sync'`

- [ ] **Step 3: Create edge_sync.py**

```python
"""Edge-to-cloud delivery: store-and-forward queue, pluggable alert sinks,
LAN life-safety alarm, and self-health heartbeat.

Trust boundary: the edge only pushes outward (cloud) or alarms on-prem facility
infrastructure (LAN). Nothing here serves data to workers/families directly and
nothing opens an inbound port on the edge.
"""

from __future__ import annotations

import enum
import json
import time
import urllib.request


class Severity(enum.IntEnum):
    ROUTINE = 0
    CRITICAL = 1


CRITICAL_CLASSES = {"fall", "aggression"}
CRITICAL_EVENTS = {"unresponsive", "abnormal_vitals"}


def classify_severity(event: dict) -> Severity:
    """Critical = life-safety (fall, unresponsiveness, severe abnormal vitals)."""
    if event.get("class") in CRITICAL_CLASSES:
        return Severity.CRITICAL
    if event.get("event") in CRITICAL_EVENTS:
        return Severity.CRITICAL
    return Severity.ROUTINE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestSeverity -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add edge_sync.py tests/test_pipeline.py
git commit -m "feat: edge_sync severity classification"
```

---

### Task 2: Durable store-and-forward queue

**Files:**
- Modify: `edge_sync.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestStoreAndForwardQueue:
    def test_enqueue_persists_to_disk(self, tmp_path):
        from edge_sync import StoreAndForwardQueue
        q = StoreAndForwardQueue(str(tmp_path / "q.jsonl"))
        q.enqueue({"class": "fall", "id": 1})
        q2 = StoreAndForwardQueue(str(tmp_path / "q.jsonl"))  # reload
        assert len(q2) == 1

    def test_drain_calls_sender_and_clears(self, tmp_path):
        from edge_sync import StoreAndForwardQueue
        q = StoreAndForwardQueue(str(tmp_path / "q.jsonl"))
        q.enqueue({"id": 1})
        q.enqueue({"id": 2})
        sent = []
        q.drain(lambda rec: sent.append(rec) or True)  # sender returns True=ok
        assert [r["id"] for r in sent] == [1, 2]
        assert len(q) == 0

    def test_drain_stops_on_failure_and_keeps_remainder(self, tmp_path):
        from edge_sync import StoreAndForwardQueue
        q = StoreAndForwardQueue(str(tmp_path / "q.jsonl"))
        q.enqueue({"id": 1}); q.enqueue({"id": 2}); q.enqueue({"id": 3})
        def sender(rec):
            return rec["id"] == 1  # only first succeeds
        q.drain(sender)
        assert len(q) == 2  # ids 2 and 3 remain for next reconnect

    def test_bounded_ring_drops_oldest_routine_first(self, tmp_path):
        from edge_sync import StoreAndForwardQueue, Severity
        q = StoreAndForwardQueue(str(tmp_path / "q.jsonl"), max_len=2)
        q.enqueue({"id": 1}, severity=Severity.ROUTINE)
        q.enqueue({"id": 2}, severity=Severity.CRITICAL)
        q.enqueue({"id": 3}, severity=Severity.ROUTINE)  # over cap
        ids = {r["id"] for r in q.records()}
        assert 2 in ids  # critical retained
        assert len(q) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestStoreAndForwardQueue -v`
Expected: FAIL with `ImportError: cannot import name 'StoreAndForwardQueue'`

- [ ] **Step 3: Implement in edge_sync.py**

Append:

```python
class StoreAndForwardQueue:
    """Durable local queue (JSONL). Survives restarts; drains in FIFO order;
    stops draining on first send failure so ordering and at-least-once delivery
    hold. Bounded: when over capacity, drop oldest ROUTINE first, never a
    CRITICAL while any ROUTINE remains.
    """

    def __init__(self, path: str, max_len: int = 10_000):
        self.path = path
        self.max_len = max_len
        self._items = []  # list of (severity:int, record:dict)
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        self._items.append((obj.get("_sev", 0), obj.get("rec", obj)))
        except FileNotFoundError:
            pass

    def _flush(self):
        with open(self.path, "w") as f:
            for sev, rec in self._items:
                f.write(json.dumps({"_sev": int(sev), "rec": rec}) + "\n")

    def enqueue(self, record: dict, severity=0):
        self._items.append((int(severity), record))
        if len(self._items) > self.max_len:
            self._evict_one()
        self._flush()

    def _evict_one(self):
        for i, (sev, _) in enumerate(self._items):
            if sev == 0:  # oldest routine
                self._items.pop(i)
                return
        self._items.pop(0)  # all critical -> drop oldest

    def drain(self, sender):
        """Send queued records in order via sender(record)->bool. Stop at first
        failure; keep the remainder."""
        remaining = []
        stopped = False
        for sev, rec in self._items:
            if stopped:
                remaining.append((sev, rec))
                continue
            ok = False
            try:
                ok = bool(sender(rec))
            except Exception:
                ok = False
            if not ok:
                stopped = True
                remaining.append((sev, rec))
        self._items = remaining
        self._flush()

    def records(self):
        return [rec for _, rec in self._items]

    def __len__(self):
        return len(self._items)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestStoreAndForwardQueue -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add edge_sync.py tests/test_pipeline.py
git commit -m "feat: durable store-and-forward queue with severity-aware eviction"
```

---

### Task 3: Sink interface + ConsoleSink

**Files:**
- Modify: `edge_sync.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestSinks:
    def test_console_sink_always_succeeds(self, capsys):
        from edge_sync import ConsoleSink
        sink = ConsoleSink()
        ok = sink.send({"class": "fall", "track_id": 2, "timestamp": 1.0})
        assert ok is True
        assert "fall" in capsys.readouterr().out.lower()

    def test_sink_accepts_filters_by_min_severity(self):
        from edge_sync import ConsoleSink, Severity
        sink = ConsoleSink(min_severity=Severity.CRITICAL)
        assert sink.accepts(Severity.CRITICAL) is True
        assert sink.accepts(Severity.ROUTINE) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestSinks -v`
Expected: FAIL with `ImportError: cannot import name 'ConsoleSink'`

- [ ] **Step 3: Implement in edge_sync.py**

Append:

```python
class AlertSink:
    """Base sink. accepts() gates by severity; send() delivers, returns success."""

    def __init__(self, min_severity=Severity.ROUTINE):
        self.min_severity = min_severity

    def accepts(self, severity) -> bool:
        return int(severity) >= int(self.min_severity)

    def send(self, event: dict) -> bool:
        raise NotImplementedError


class ConsoleSink(AlertSink):
    def send(self, event: dict) -> bool:
        track = event.get("track_id")
        label = f" (Person {track})" if track is not None else ""
        name = event.get("class") or event.get("event") or "event"
        print(f"[ALERT] {name.upper()}{label} @ {event.get('timestamp', 0):.1f}s")
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestSinks -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add edge_sync.py tests/test_pipeline.py
git commit -m "feat: AlertSink interface + ConsoleSink"
```

---

### Task 4: CloudSink (HTTP push via the queue)

**Files:**
- Modify: `edge_sync.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestCloudSink:
    def test_send_posts_and_returns_true_on_success(self, monkeypatch):
        from edge_sync import CloudSink
        calls = []
        monkeypatch.setattr("edge_sync._http_post",
                            lambda url, payload, timeout=5: calls.append((url, payload)) or True)
        sink = CloudSink("https://cloud.example/api")
        assert sink.send({"class": "fall"}) is True
        assert calls and calls[0][0] == "https://cloud.example/api"

    def test_send_returns_false_on_failure(self, monkeypatch):
        from edge_sync import CloudSink
        def boom(url, payload, timeout=5):
            raise OSError("network down")
        monkeypatch.setattr("edge_sync._http_post", boom)
        sink = CloudSink("https://cloud.example/api")
        assert sink.send({"class": "fall"}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestCloudSink -v`
Expected: FAIL with `ImportError: cannot import name 'CloudSink'`

- [ ] **Step 3: Implement in edge_sync.py**

Append:

```python
def _http_post(url: str, payload: dict, timeout: int = 5) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return 200 <= resp.status < 300


class CloudSink(AlertSink):
    """Push-only delivery to the cloud. Returns False on any network error so
    the caller can keep the record queued for retry."""

    def __init__(self, url: str, min_severity=Severity.ROUTINE, timeout: int = 5):
        super().__init__(min_severity)
        self.url = url
        self.timeout = timeout

    def send(self, event: dict) -> bool:
        try:
            return _http_post(self.url, event, timeout=self.timeout)
        except Exception:
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestCloudSink -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add edge_sync.py tests/test_pipeline.py
git commit -m "feat: CloudSink — push-only HTTP delivery with failure signaling"
```

---

### Task 5: LANAlarmSink (on-prem life-safety, critical only)

**Files:**
- Modify: `edge_sync.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestLANAlarmSink:
    def test_only_accepts_critical(self):
        from edge_sync import LANAlarmSink, Severity
        sink = LANAlarmSink("http://10.0.0.5/alarm")
        assert sink.accepts(Severity.CRITICAL) is True
        assert sink.accepts(Severity.ROUTINE) is False

    def test_fires_independent_of_cloud(self, monkeypatch):
        from edge_sync import LANAlarmSink
        calls = []
        monkeypatch.setattr("edge_sync._http_post",
                            lambda url, payload, timeout=3: calls.append(url) or True)
        sink = LANAlarmSink("http://10.0.0.5/alarm")
        assert sink.send({"class": "fall"}) is True
        assert calls == ["http://10.0.0.5/alarm"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestLANAlarmSink -v`
Expected: FAIL with `ImportError: cannot import name 'LANAlarmSink'`

- [ ] **Step 3: Implement in edge_sync.py**

Append:

```python
class LANAlarmSink(AlertSink):
    """One-way life-safety alarm to on-prem facility infrastructure over the
    LAN. Critical events only. Independent of the cloud, so it fires during a
    cloud outage. This is an alarm to facility systems, not a data-serving UI."""

    def __init__(self, url: str, timeout: int = 3):
        super().__init__(min_severity=Severity.CRITICAL)
        self.url = url
        self.timeout = timeout

    def send(self, event: dict) -> bool:
        payload = {"alarm": event.get("class") or event.get("event"),
                   "track_id": event.get("track_id"),
                   "timestamp": event.get("timestamp")}
        try:
            return _http_post(self.url, payload, timeout=self.timeout)
        except Exception:
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestLANAlarmSink -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add edge_sync.py tests/test_pipeline.py
git commit -m "feat: LANAlarmSink — on-prem critical-only life-safety alarm"
```

---

### Task 6: Heartbeat + AlertRouter (fan-out + queue integration)

**Files:**
- Modify: `edge_sync.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestAlertRouter:
    def test_fans_out_to_accepting_sinks(self, tmp_path):
        from edge_sync import AlertRouter, AlertSink, Severity, StoreAndForwardQueue

        class Recorder(AlertSink):
            def __init__(self, min_sev=Severity.ROUTINE):
                super().__init__(min_sev); self.got = []
            def send(self, e): self.got.append(e); return True

        routine_sink = Recorder(Severity.ROUTINE)
        critical_sink = Recorder(Severity.CRITICAL)
        q = StoreAndForwardQueue(str(tmp_path / "q.jsonl"))
        router = AlertRouter(sinks=[routine_sink, critical_sink], queue=q)

        router.dispatch({"class": "eating"})      # routine
        router.dispatch({"class": "fall"})        # critical
        assert len(routine_sink.got) == 2         # routine sink sees both
        assert len(critical_sink.got) == 1        # critical sink sees only fall

    def test_failed_cloud_send_queues_for_retry(self, tmp_path):
        from edge_sync import AlertRouter, AlertSink, Severity, StoreAndForwardQueue

        class Failing(AlertSink):
            queue_on_fail = True
            def send(self, e): return False

        q = StoreAndForwardQueue(str(tmp_path / "q.jsonl"))
        router = AlertRouter(sinks=[Failing()], queue=q)
        router.dispatch({"class": "fall"})
        assert len(q) == 1  # failed send was queued

    def test_heartbeat_payload(self):
        from edge_sync import make_heartbeat
        hb = make_heartbeat(device_id="cam-01", timestamp=123.0)
        assert hb["device_id"] == "cam-01"
        assert hb["event"] == "heartbeat"
        assert hb["timestamp"] == 123.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestAlertRouter -v`
Expected: FAIL with `ImportError: cannot import name 'AlertRouter'`

- [ ] **Step 3: Implement in edge_sync.py**

Append:

```python
def make_heartbeat(device_id: str, timestamp: float) -> dict:
    return {"event": "heartbeat", "device_id": device_id, "timestamp": timestamp}


class AlertRouter:
    """Fan out an event to all accepting sinks by severity. A sink with
    `queue_on_fail = True` whose send() returns False causes the record to be
    enqueued for later retry (store-and-forward)."""

    def __init__(self, sinks, queue=None):
        self.sinks = sinks
        self.queue = queue

    def dispatch(self, event: dict):
        severity = classify_severity(event)
        for sink in self.sinks:
            if not sink.accepts(severity):
                continue
            ok = sink.send(event)
            if not ok and getattr(sink, "queue_on_fail", False) and self.queue is not None:
                self.queue.enqueue(event, severity=severity)

    def flush_queue(self, sender):
        """Drain the store-and-forward queue through `sender` on reconnect."""
        if self.queue is not None:
            self.queue.drain(sender)
```

Set `queue_on_fail = True` on `CloudSink` (it is the sink whose failures must be
retried). Add the class attribute to `CloudSink`:

```python
class CloudSink(AlertSink):
    queue_on_fail = True
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestAlertRouter -v`
Expected: 3 passed

- [ ] **Step 5: Run the FULL suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add edge_sync.py tests/test_pipeline.py
git commit -m "feat: AlertRouter fan-out + queue-on-fail + heartbeat"
```

---

### Task 7: Refactor AlertDispatcher to delegate to sinks

**Files:**
- Modify: `stream_detect.py` (`AlertDispatcher` ~line 673)
- Test: `tests/test_pipeline.py`

Goal: preserve current behavior (JSONL log + console + optional webhook) while
routing through `AlertRouter`, so the new sinks compose. The existing
`dispatch(event)` signature stays.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestAlertDispatcherRouter:
    def test_dispatch_still_logs_jsonl(self, tmp_path):
        from stream_detect import AlertDispatcher
        d = AlertDispatcher(output_dir=str(tmp_path))
        d.dispatch({"class": "fall", "confidence": 0.9, "timestamp": 1.0})
        log = (tmp_path / "event_log.jsonl").read_text().strip()
        assert "fall" in log

    def test_dispatch_routes_to_extra_sinks(self, tmp_path):
        from stream_detect import AlertDispatcher
        from edge_sync import AlertSink, Severity

        class Spy(AlertSink):
            def __init__(self): super().__init__(Severity.CRITICAL); self.got = []
            def send(self, e): self.got.append(e); return True

        spy = Spy()
        d = AlertDispatcher(output_dir=str(tmp_path), sinks=[spy])
        d.dispatch({"class": "fall", "confidence": 0.9, "timestamp": 1.0})
        d.dispatch({"class": "eating", "confidence": 0.9, "timestamp": 2.0})
        assert len(spy.got) == 1  # only the critical fall reached the critical sink
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestAlertDispatcherRouter -v`
Expected: FAIL (`AlertDispatcher` has no `sinks` parameter)

- [ ] **Step 3: Refactor AlertDispatcher**

Replace `AlertDispatcher.__init__` and `dispatch` in `stream_detect.py` with:

```python
    def __init__(self, output_dir: str = "events", webhook_url: str | None = None,
                 sinks=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "event_log.jsonl"
        self.webhook_url = webhook_url
        from edge_sync import AlertRouter
        self.router = AlertRouter(sinks=sinks or [])

    def dispatch(self, event: dict):
        """Log event (JSONL), print console alert, route to sinks, optional webhook."""
        event["detected_at"] = datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        print(f"\n{'='*60}")
        track_label = f" (Person {event['track_id']})" if "track_id" in event else ""
        name = event.get("class") or event.get("event") or "event"
        print(f"  EVENT DETECTED: {name.upper()}{track_label}")
        if "confidence" in event:
            print(f"  Confidence: {event['confidence']:.1%}")
        print(f"  Video time: {event.get('timestamp', 0):.1f}s")
        if event.get("clip_path"):
            print(f"  Clip saved: {event['clip_path']}")
        print(f"{'='*60}\n")

        self.router.dispatch(event)

        if self.webhook_url:
            self._send_webhook(event)
```

Keep `_send_webhook` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestAlertDispatcherRouter -v`
Expected: 2 passed

- [ ] **Step 5: Full suite (regression — confirm existing alert behavior intact)**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add stream_detect.py tests/test_pipeline.py
git commit -m "refactor: AlertDispatcher routes through AlertRouter sinks (behavior-preserving)"
```

---

### Task 8: CLI wiring + reconnect drain

**Files:**
- Modify: `stream_detect.py` (`main()` argparse; `run_pipeline` dispatcher construction + periodic drain)

- [ ] **Step 1: Add CLI flags in main()**

```python
    parser.add_argument("--cloud-url", default=None,
                        help="Cloud ingest URL (push-only; edge never serves UIs)")
    parser.add_argument("--lan-alarm-url", default=None,
                        help="On-prem LAN alarm endpoint for critical events")
    parser.add_argument("--queue-path", default="events/forward_queue.jsonl",
                        help="Durable store-and-forward queue path")
    parser.add_argument("--device-id", default="edge-0",
                        help="Device id for heartbeat")
    parser.add_argument("--heartbeat-interval", type=float, default=30.0,
                        help="Seconds between edge self-health heartbeats")
```

- [ ] **Step 2: Build sinks + dispatcher in run_pipeline**

Where `AlertDispatcher` is currently constructed in `run_pipeline`, replace with:

```python
    from edge_sync import ConsoleSink, CloudSink, LANAlarmSink, StoreAndForwardQueue, Severity
    sinks = []
    queue = StoreAndForwardQueue(args.queue_path)
    if args.cloud_url:
        sinks.append(CloudSink(args.cloud_url))
    if args.lan_alarm_url:
        sinks.append(LANAlarmSink(args.lan_alarm_url))
    alerter = AlertDispatcher(output_dir=args.output_dir,
                              webhook_url=args.webhook, sinks=sinks)
    alerter.router.queue = queue
```

- [ ] **Step 3: Periodic reconnect drain + heartbeat**

Once per frame (or on a timer) in the loop, attempt to drain the queue and send
a heartbeat through the cloud sink when configured:

```python
            if args.cloud_url and frame_idx % int(max(1, args.heartbeat_interval * source.fps)) == 0:
                from edge_sync import make_heartbeat
                cloud = next((s for s in sinks if s.__class__.__name__ == "CloudSink"), None)
                if cloud is not None:
                    alerter.router.flush_queue(cloud.send)
                    cloud.send(make_heartbeat(args.device_id, timestamp))
```

(`frame_idx` exists if Plan 1 is present; if running Plan 2 standalone, add a
`frame_idx = 0` / `frame_idx += 1` counter in the loop.)

- [ ] **Step 4: Verify CLI + imports**

Run: `.venv/bin/python stream_detect.py --help 2>&1 | grep -E -- "--cloud-url|--lan-alarm-url|--queue-path"`
Expected: shows the three flags.

Run: `.venv/bin/python -c "import stream_detect, edge_sync; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add stream_detect.py
git commit -m "feat: wire cloud/LAN sinks, store-and-forward drain, heartbeat into pipeline"
```

---

### Task 9: Final verification

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v 2>&1 | tail -20`
Expected: all pass, zero failures.

- [ ] **Step 2: Trust-boundary self-check (manual read)**

Confirm by reading `edge_sync.py`: no sink opens a listening socket; all network
calls are outbound `urllib.request.urlopen` POSTs (to cloud or LAN endpoints).
The edge never binds an inbound port.

- [ ] **Step 3: Push**

```bash
git push origin HEAD
```
