# ROADMAP

Actionable work queue for the detection service and research ideas.

- **Service hardening (S1–S10)** — small, high-leverage fixes to the streaming
  service logic, found by review of [rpc_server.py](rpc_server.py),
  [rpc_client.py](rpc_client.py), [consumer.py](consumer.py),
  [functions/streamz_tools.py](functions/streamz_tools.py),
  [functions/model_persistence.py](functions/model_persistence.py).
- **Research ideas (I1–I7)** — mapped to code and the ESwA paper in
  [IDEAS.md](IDEAS.md).

## Priority table

| Code | Item | Type | Impact | Effort |
|------|------|------|--------|--------|
| S1 | One bad message kills the service | stability | High | Low |
| S2 | File sink never flushed/closed in production | stability | High | Low |
| S3 | `consumer.py` NameError when encryption is off | bug | High | Trivial |
| S4 | Email alerting breaks when encryption is enabled | bug | High | Low |
| S5 | Brittle stop detection in `run()` | stability | Medium | Low |
| S6 | MQTT publish: no reconnect, errors ignored | stability | Medium | Low |
| S7 | Kafka `group.id` silently overridden | usability | Medium | Trivial |
| S8 | Recovered model silently ignores current config | usability | Medium | Low |
| S9 | Stray no-op `MQTTMessage()` | cleanup | Low | Trivial |
| S10 | Fragile encryption-detection heuristic in consumer | cleanup | Low | Trivial |

Suggested order: **S1 → S2 → S3 → S4** (an afternoon combined; removes the main
ways the service falls over or lies to you), then S5–S8, trivia S9–S10 whenever
the file is open anyway. S4 interacts with [IDEAS.md → I3](IDEAS.md#i3--fix-encryption-returns-limits-as-liststr)
(stringly-typed limits after decryption).

---

## S1 — One bad message kills the entire service

**Problem:** the custom `map` node calls `self.stop(); self.destroy()` and
re-raises on *any* exception:
[functions/streamz_tools.py:21-28](functions/streamz_tools.py#L21-L28).
Combined with:

- `preprocess` returning `None` for unrecognized input types
  ([rpc_server.py:193](rpc_server.py#L193)), which then crashes
  `fit_transform` on `x["data"]`,
- unguarded `float(x.payload)` / `float(x.decode(...))` parses
  ([rpc_server.py:183](rpc_server.py#L183),
  [rpc_server.py:191](rpc_server.py#L191)),

a single malformed MQTT payload (e.g. a retained non-numeric message) takes
down the whole detector.

**Fix:**
1. In `map.update`, log-and-skip the failing message (`return` instead of
   stop/destroy), or make the behavior configurable
   (`on_error="skip" | "raise"`).
2. Add `.filter(lambda x: x is not None)` after the preprocess map at
   [rpc_server.py:512](rpc_server.py#L512).
3. Wrap the `float(...)` parses in `preprocess` so a parse failure returns
   `None` (handled by the filter) with a warning log.

**Acceptance:** feeding a non-numeric payload into a running pipeline logs a
warning and the next valid message is still processed.

## S2 — File sink output never flushed/closed in production

**Problem:** `get_sink` opens the output file and stashes it in the global
`open_files` ([rpc_server.py:392-394](rpc_server.py#L392-L394)), but files are
closed only in the *debug* branch of `run()`
([rpc_server.py:432-433](rpc_server.py#L432-L433)). In normal operation,
buffered JSON lines may never reach disk — which is exactly what
`consumer.py`'s `query_file` ([consumer.py:63-106](consumer.py#L63-L106))
reads.

**Fix:**
1. `print(json.dumps(x), file=f, flush=True)` in `dump_to_file`
   ([rpc_server.py:247-248](rpc_server.py#L247-L248)).
2. Close all `open_files` in the `finally` block of `start()`
   ([rpc_server.py:533-538](rpc_server.py#L533-L538)), not only in debug mode.

**Acceptance:** while the service runs, `tail -f` on the output file shows each
result as it is produced; no data loss on SIGTERM.

## S3 — `consumer.py` NameError when encryption is off

**Problem:** `receiver` is bound only inside
`if "key_path" in config["setup"]`
([consumer.py:143-144](consumer.py#L143-L144)) but is unconditionally passed at
[consumer.py:148](consumer.py#L148) → `NameError` for any unencrypted setup.

**Fix:** initialize `receiver = None` before the conditional; in `query_file`
([consumer.py:82-87](consumer.py#L82-L87)) and `on_message`
([consumer.py:51-58](consumer.py#L51-L58)), skip decryption when receiver is
`None`.

**Acceptance:** consumer runs against a plaintext output file / topic without
a key path configured.

## S4 — Email alerting breaks when encryption is enabled

**Problem:** the email sink is attached *after* the sign→encrypt→decode maps
([rpc_server.py:517-531](rpc_server.py#L517-L531)), so `send_anomaly_email`
computes `xs[1]["anomaly"] - xs[0]["anomaly"]`
([rpc_server.py:256](rpc_server.py#L256)) on encrypted *string* values →
`TypeError`. Per S1 semantics today, this kills the service on the first
anomaly transition — the worst possible moment.

**Fix:** attach the email branch to the plaintext detector node (right after
`fit_transform`, before the crypto maps), e.g. keep a `plain = detector`
reference at [rpc_server.py:512-515](rpc_server.py#L512-L515) and hang the
`sliding_window(2).sink(...)` off `plain`.

**Cross-ref:** root cause of stringly-typed payloads is
[IDEAS.md → I3](IDEAS.md#i3--fix-encryption-returns-limits-as-liststr).

**Acceptance:** with `key_path` and email both configured, an anomaly
transition sends an email and the service keeps running.

## S5 — Brittle stop detection in `run()`

**Problem:** the polling loop probes
`source.upstreams[0].upstreams[0].stopped`
([rpc_server.py:439-446](rpc_server.py#L439-L446)) — hardcoded to the exact
depth of the MQTT `accumulate().filter()` chain built in `get_source`
([rpc_server.py:344-348](rpc_server.py#L344-L348)). Any pipeline change breaks
shutdown detection.

**Fix:** in `get_source`, keep a reference to the raw source node before
wrapping (e.g. return `(raw_source, wrapped)` or set `self._raw_source`) and
poll that single object in `run()`.

## S6 — MQTT publish: no reconnect, errors ignored

**Problem:** the sink client is created once
([functions/streamz_tools.py:128-135](functions/streamz_tools.py#L128-L135));
after a broker drop, every `publish()` fails silently (paho returns an error
code, it does not raise). The
`# TODO: wait on successful delivery` at
[functions/streamz_tools.py:136](functions/streamz_tools.py#L136) is exactly
this.

**Fix:** check the publish result — on
`result.rc != mqtt.MQTT_ERR_SUCCESS`, call `self.client.reconnect()` and retry
once; optionally `result.wait_for_publish(timeout=...)` for QoS > 0. Apply to
both `client.publish` call sites
([functions/streamz_tools.py:138](functions/streamz_tools.py#L138),
[:140-167](functions/streamz_tools.py#L140-L167)).

## S7 — Kafka `group.id` silently overridden

**Problem:** `{**config, "group.id": "detection_service"}`
([rpc_server.py:350-353](rpc_server.py#L350-L353)) clobbers a user-supplied
group id — confusing when running multiple consumers; the docstring example
([rpc_server.py:311-312](rpc_server.py#L311-L312)) even suggests passing one.

**Fix:** flip the merge so user config wins:
`{"group.id": "detection_service", **config}`.

## S8 — Recovered model silently ignores current config

**Problem:** `load_model` restores a pickle and skips model construction
entirely ([rpc_server.py:490-508](rpc_server.py#L490-L508)), so changing
`threshold` / `t_e` / `t_a` / `t_g` in config does nothing when a recovery
file exists. Additionally, `save_model` writes a new timestamped pickle on
every shutdown with no retention
([functions/model_persistence.py:48-55](functions/model_persistence.py#L48-L55)).

**Fix:**
1. After a successful `load_model`, compare the recovered model's
   `threshold` / `t_e` / `t_a` / `grace_period` against
   `expand_model_params(...)` output and `logger.warning` on mismatch (or
   re-apply the configurable ones).
2. Optional: prune recovery pickles to the last N in `save_model`.

## S9 — Stray no-op `MQTTMessage()`

Dead statement creating and discarding an object:
[functions/streamz_tools.py:229](functions/streamz_tools.py#L229). Delete.

## S10 — Fragile encryption-detection heuristic in consumer

`query_file` decides an item is encrypted via `not item["time"].isascii()`
([consumer.py:82](consumer.py#L82)). Checking `"signature" in item` is direct
and self-describing. (Becomes moot once
[IDEAS.md → I3](IDEAS.md#i3--fix-encryption-returns-limits-as-liststr) gives
the payload a proper serialization boundary.)

---

## Research ideas

Tracked in [IDEAS.md](IDEAS.md) with full code/paper cross-references:

| Code | Idea | Value | Effort |
|------|------|-------|--------|
| I3 | Fix encryption returns limits as list/str | High | Low |
| I4 | Option to provide physical limits | High | Medium |
| I2 | Make part of `GaussianScorer` public | Medium | Low |
| I1 | Protect anomaly detector with ThresholdFilter | Medium | Medium |
| I6 | TSB-AD benchmark comparison | High | High |
| I7 | Sensor fault diagnosis taxonomy | High | Very High |
| I5 | Compare with Reunanen et al. 2020 | Medium | High |
