"""Control-rate benchmark for the client <-> GR00T policy server pipeline.

Emulates eval_g1_sim_groot's 30 Hz async control loop with fake observations —
no Isaac Sim needed — and reports whether the loop holds the target rate and
whether the action queue ever underruns (the condition that makes the real arm
hold its last command, i.e. a hitch).

Run on the DESKTOP through the SSH tunnel to include real network latency
(needs only pyzmq/msgpack/msgpack-numpy/numpy/torch, same as the client):

    python test_control_rate.py --server 127.0.0.1:5555 \
        --actions_per_chunk 16 --frequency 30 --seconds 30
"""

import argparse
import sys
import threading
import time

import numpy as np
import torch

from gr00t_sim_policy_client import Gr00tSimPolicyClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:5555")
    ap.add_argument("--actions_per_chunk", type=int, default=16)
    ap.add_argument("--chunk_size_threshold", type=float, default=0.5)
    ap.add_argument("--frequency", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    client = Gr00tSimPolicyClient(
        server_address=args.server,
        fps=args.frequency,
        chunk_size_threshold=args.chunk_size_threshold,
        actions_per_chunk=args.actions_per_chunk,
    )
    client.start()
    receiver = threading.Thread(target=client.receive_actions_loop, daemon=True)
    receiver.start()

    rng = np.random.default_rng(0)
    observation = {
        "observation.state": torch.zeros(16, dtype=torch.float32),
        "task": "Pick up the red cube and place it in the yellow region.",
    }
    for cam in client.video_keys:
        observation[f"observation.images.{cam}"] = torch.from_numpy(
            rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        )

    dt = 1.0 / args.frequency
    n_ticks = int(args.seconds * args.frequency)
    tick_busy: list[float] = []
    tick_starts: list[float] = []
    misses_after_warmup = 0
    warmup_ticks = None  # ticks until the first action arrived

    print(f"Running {n_ticks} ticks @ {args.frequency:.0f} Hz against {args.server} ...")
    bench_start = time.perf_counter()
    for i in range(n_ticks):
        t0 = time.perf_counter()
        tick_starts.append(t0)

        if client.ready_to_send_observation():
            client.send_observation(dict(observation), timestep=client.current_timestep())

        action = client.pop_action()
        if action is not None:
            if warmup_ticks is None:
                warmup_ticks = i
        elif warmup_ticks is not None:
            misses_after_warmup += 1

        tick_busy.append(time.perf_counter() - t0)
        time.sleep(max(0, dt - (time.perf_counter() - t0)))
    total_elapsed = time.perf_counter() - bench_start

    client.stop()

    if warmup_ticks is None:
        print("FAIL: no action ever arrived — server unreachable or inference broken")
        return 1

    starts = np.array(tick_starts)
    periods = np.diff(starts)
    busy = np.array(tick_busy)
    snap = client.timing_snapshot()
    achieved_hz = (n_ticks - 1) / (starts[-1] - starts[0])
    active_ticks = n_ticks - warmup_ticks - 1

    print()
    print(f"== control loop (target {args.frequency:.0f} Hz, {total_elapsed:.1f}s) ==")
    print(f"achieved rate     : {achieved_hz:.2f} Hz")
    print(
        f"tick period       : p50 {np.percentile(periods, 50) * 1e3:.1f} ms | "
        f"p95 {np.percentile(periods, 95) * 1e3:.1f} ms | max {periods.max() * 1e3:.1f} ms"
    )
    print(
        f"tick busy time    : p95 {np.percentile(busy, 95) * 1e3:.1f} ms "
        f"(budget {1e3 * (1 / args.frequency):.1f} ms)"
    )
    print()
    print("== action pipeline ==")
    print(f"warm-up           : {warmup_ticks} ticks until first action")
    print(f"queue underruns   : {misses_after_warmup} / {active_ticks} active ticks")
    print(
        f"inference latency : p50 {snap.get('infer_ms_p50', float('nan')):.0f} ms | "
        f"p95 {snap.get('infer_ms_p95', float('nan')):.0f} ms | "
        f"max {snap.get('infer_ms_max', float('nan')):.0f} ms  ({snap['replans']:.0f} replans)"
    )
    print(f"replan interval   : p50 {snap.get('replan_interval_ms_p50', float('nan')):.0f} ms")

    # Verdict: the loop must hold the rate and, once warmed up, an action must
    # be available every tick. Latency headroom = time to refill before the
    # buffered chunk drains below the threshold.
    budget_ms = 1e3 * (1 - args.chunk_size_threshold) * args.actions_per_chunk / args.frequency
    ok_rate = abs(achieved_hz - args.frequency) / args.frequency < 0.02
    ok_misses = misses_after_warmup == 0
    ok_latency = snap.get("infer_ms_p95", float("inf")) < budget_ms
    print()
    print(f"rate within 2%    : {'OK' if ok_rate else 'FAIL'}")
    print(f"zero underruns    : {'OK' if ok_misses else 'FAIL'}")
    print(
        f"latency headroom  : {'OK' if ok_latency else 'MARGINAL'} "
        f"(p95 vs {budget_ms:.0f} ms refill budget)"
    )
    print()
    if ok_rate and ok_misses:
        print("CONTROL RATE TEST PASSED")
        return 0
    print("CONTROL RATE TEST FAILED — see §5 of the README for tuning knobs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
