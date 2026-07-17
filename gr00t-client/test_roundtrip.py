"""End-to-end round-trip test: fake LeRobot observation -> Gr00tSimPolicyClient
-> Isaac-GR00T PolicyServer -> action chunk.

Run on the GPU server with the Isaac-GR00T venv, with the policy server already
listening (see serve_variant.sh):

    /NHNHOME/WORKSPACE/chan/Isaac-GR00T/.venv/bin/python test_roundtrip.py \
        --port 5555 --horizon 16
"""

import argparse
import sys
import time

from gr00t_sim_policy_client import Gr00tSimPolicyClient
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument(
        "--horizon", type=int, default=16, help="expected action horizon of the served variant"
    )
    args = ap.parse_args()

    client = Gr00tSimPolicyClient(
        server_address=f"{args.host}:{args.port}",
        fps=30.0,
        actions_per_chunk=args.horizon,
    )
    client.start()
    print(f"server modality config OK: video={client.video_keys} action={client.action_keys}")

    # Fake LeRobot-shaped observation, matching what eval_g1_sim_groot assembles.
    observation = {
        "observation.state": torch.zeros(16, dtype=torch.float32),
        "task": "Pick up the red cube and place it in the yellow region.",
    }
    for cam in client.video_keys:
        observation[f"observation.images.{cam}"] = torch.randint(
            0, 255, (480, 640, 3), dtype=torch.uint8
        )

    assert client.ready_to_send_observation()
    assert client.send_observation(observation, timestep=0)

    # Drive one inference round trip synchronously (no background thread needed).
    converted, timestep, t_0 = client.observation_queue.get(timeout=1.0)
    t_start = time.perf_counter()
    action = client.client.get_action(converted)
    latency = time.perf_counter() - t_start
    chunk = client._chunk_from_response(action)

    print(f"inference latency: {latency * 1e3:.0f} ms")
    print(f"action keys: {sorted(action.keys())}")
    print(f"chunk shape: {tuple(chunk.shape)} dtype={chunk.dtype}")

    assert chunk.shape == (args.horizon, 16), (
        f"expected ({args.horizon}, 16), got {tuple(chunk.shape)}"
    )
    assert torch.isfinite(chunk).all(), "chunk contains non-finite values"
    grippers = chunk[:, 14:16]
    print(f"arm action range:     [{chunk[:, :14].min():.3f}, {chunk[:, :14].max():.3f}] rad")
    print(
        f"gripper action range: [{grippers.min():.3f}, {grippers.max():.3f}] (dataset units 0~5.4)"
    )

    # Exercise the queue path as eval_g1_sim_groot would.
    client._replace_action_queue(
        [  # mimic receive_actions_loop
            __import__("gr00t_sim_policy_client").TimedAction(
                timestamp=t_0 + i / 30.0, timestep=timestep + i, action=chunk[i]
            )
            for i in range(chunk.shape[0])
        ]
    )
    popped = client.pop_action()
    assert popped is not None and popped.shape == (16,)
    print(f"pop_action OK: first action = {np.round(popped.numpy(), 3)}")

    client.stop()
    print("ROUNDTRIP TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
