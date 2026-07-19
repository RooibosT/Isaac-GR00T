"""Drop-in replacement for unitree_lerobot's RemoteSimPolicyClient that talks to
an Isaac-GR00T PolicyServer (``gr00t/eval/run_gr00t_server.py``) instead of the
LeRobot gRPC server.

Same public interface as RemoteSimPolicyClient:
    start / stop / reset / ready_to_send_observation / send_observation /
    receive_actions_loop / pop_action / current_timestep

Observation in  (LeRobot-shaped, from eval_g1_sim_*):
    {"observation.images.<cam>": torch.uint8 (H, W, 3),
     "observation.state": torch.float32 (16,),
     "task": str}
Wire out (Isaac-GR00T Gr00tPolicy format):
    {"video": {<cam>: np.uint8 (1, 1, H, W, 3)},
     "state": {<group>: np.float32 (1, 1, D)},
     "language": {<lang_key>: [[task]]}}
Actions returned by the server ({<group>: (1, T, D) float32}) are concatenated
in modality-config key order into per-step (16,) tensors.
"""

from dataclasses import dataclass
from collections import deque
import logging
from queue import Queue
import threading
import time
from typing import Any

from gr00t_transport import Gr00tRemoteClient
import numpy as np
import torch


logger = logging.getLogger(__name__)

# state/action layout of the G1 + Dex1 dataset (meta/modality.json).
DEFAULT_STATE_LAYOUT: dict[str, tuple[int, int]] = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_gripper": (14, 15),
    "right_gripper": (15, 16),
}


@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: torch.Tensor

    def get_timestep(self) -> int:
        return self.timestep

    def get_action(self) -> torch.Tensor:
        return self.action


class Gr00tSimPolicyClient:
    def __init__(
        self,
        server_address: str,
        fps: float,
        chunk_size_threshold: float = 0.5,
        actions_per_chunk: int = 16,
        state_layout: dict[str, tuple[int, int]] | None = None,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        host, _, port = server_address.rpartition(":")
        self.client = Gr00tRemoteClient(
            host=host or "127.0.0.1", port=int(port), timeout_ms=timeout_ms, api_token=api_token
        )
        self.fps = fps
        self.environment_dt = 1.0 / fps
        self.chunk_size_threshold = chunk_size_threshold
        self.action_chunk_size = actions_per_chunk
        self.state_layout = dict(state_layout or DEFAULT_STATE_LAYOUT)

        self.observation_queue: Queue = Queue(maxsize=1)
        self.action_queue: Queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action = -1
        self.latest_action_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # Filled by start() from the server's modality config.
        self.video_keys: list[str] = []
        self.state_keys: list[str] = list(self.state_layout.keys())
        self.action_keys: list[str] = list(self.state_layout.keys())
        self.language_key: str = "annotation.human.task_description"

        # Timing telemetry (see timing_snapshot()).
        self._stats_lock = threading.Lock()
        self._inference_latencies: deque[float] = deque(maxlen=500)
        self._replan_intervals: deque[float] = deque(maxlen=500)
        self._last_replan_time: float | None = None

    # ------------------------------------------------------------------ setup

    def start(self) -> None:
        if not self.client.ping():
            raise ConnectionError(
                f"Cannot reach GR00T policy server at {self.client.host}:{self.client.port}"
            )
        cfg = self.client.get_modality_config()
        # ModalityConfig objects arrive as plain dicts (see gr00t_transport).
        self.video_keys = list(cfg["video"]["modality_keys"])
        self.state_keys = list(cfg["state"]["modality_keys"])
        self.action_keys = list(cfg["action"]["modality_keys"])
        self.language_key = cfg["language"]["modality_keys"][0]

        missing = [k for k in self.state_keys if k not in self.state_layout]
        if missing:
            raise ValueError(
                f"State layout is missing groups {missing}; pass state_layout matching "
                f"the dataset's meta/modality.json"
            )
        self.client.reset()
        logger.info(
            "GR00T server ready. video=%s state=%s action=%s language=%s",
            self.video_keys,
            self.state_keys,
            self.action_keys,
            self.language_key,
        )

    def stop(self) -> None:
        self.shutdown_event.set()
        self.client.close()

    def reset(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()
        with self.latest_action_lock:
            self.latest_action = -1
        self.client.reset()

    # ------------------------------------------------------------- scheduling

    def ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            if self.action_queue.empty():
                return True
            return (
                self.action_queue.qsize() / max(1, self.action_chunk_size)
            ) <= self.chunk_size_threshold

    def current_timestep(self) -> int:
        with self.latest_action_lock:
            return max(self.latest_action, 0)

    # ------------------------------------------------------------ observation

    def _convert_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        task = observation.get("task") or ""

        video: dict[str, np.ndarray] = {}
        for key in self.video_keys:
            lerobot_key = f"observation.images.{key}"
            if lerobot_key not in observation:
                raise KeyError(f"Observation missing camera '{lerobot_key}'")
            frame = observation[lerobot_key]
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            frame = np.ascontiguousarray(frame).astype(np.uint8, copy=False)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Camera '{key}' must be HWC RGB, got shape {frame.shape}")
            video[key] = frame[None, None]  # (1, 1, H, W, 3)

        state_vec = observation["observation.state"]
        if isinstance(state_vec, torch.Tensor):
            state_vec = state_vec.cpu().numpy()
        state_vec = state_vec.astype(np.float32).reshape(-1)
        state: dict[str, np.ndarray] = {}
        for key in self.state_keys:
            start, end = self.state_layout[key]
            state[key] = state_vec[start:end][None, None]  # (1, 1, D)

        return {
            "video": video,
            "state": state,
            "language": {self.language_key: [[task]]},
        }

    def send_observation(self, observation: dict[str, Any], timestep: int) -> bool:
        try:
            converted = self._convert_observation(observation)
        except Exception:
            logger.exception("Failed to convert observation for GR00T server")
            return False
        if self.observation_queue.full():
            try:
                self.observation_queue.get_nowait()
            except Exception:
                pass
        self.observation_queue.put((converted, timestep, time.time()))
        return True

    # ---------------------------------------------------------------- actions

    def _chunk_from_response(self, action: dict[str, np.ndarray]) -> torch.Tensor:
        parts = []
        for key in self.action_keys:
            value = np.asarray(action[key], dtype=np.float32)
            if value.ndim == 3:  # (B, T, D) -> (T, D)
                value = value[0]
            if value.ndim == 1:  # (T,) -> (T, 1)
                value = value[:, None]
            parts.append(value)
        return torch.from_numpy(np.concatenate(parts, axis=-1))  # (T, sum(D))

    def receive_actions_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                converted, timestep, t_0 = self.observation_queue.get(timeout=0.5)
            except Exception:
                continue
            try:
                t_start = time.perf_counter()
                action = self.client.get_action(converted)
                latency = time.perf_counter() - t_start
                with self._stats_lock:
                    self._inference_latencies.append(latency)
                    if self._last_replan_time is not None:
                        self._replan_intervals.append(t_start - self._last_replan_time)
                    self._last_replan_time = t_start
                chunk = self._chunk_from_response(action)
                chunk = chunk[: self.action_chunk_size]
                dt = self.environment_dt
                timed_actions = [
                    TimedAction(timestamp=t_0 + i * dt, timestep=timestep + i, action=chunk[i])
                    for i in range(chunk.shape[0])
                ]
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._replace_action_queue(timed_actions)
            except Exception:
                if not self.shutdown_event.is_set():
                    logger.exception("GR00T inference round trip failed")
                    time.sleep(self.environment_dt)

    def pop_action(self) -> torch.Tensor | None:
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None
            timed_action = self.action_queue.get_nowait()
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        return timed_action.get_action()

    def _replace_action_queue(self, timed_actions: list[TimedAction]) -> None:
        with self.latest_action_lock:
            latest_action = self.latest_action
        future_actions = [a for a in timed_actions if a.get_timestep() > latest_action]
        with self.action_queue_lock:
            self.action_queue = Queue()
            for action in future_actions:
                self.action_queue.put(action)

    def timing_snapshot(self) -> dict[str, float]:
        """Latency/cadence stats over the recent window (for rate diagnostics)."""
        with self._stats_lock:
            lat = np.array(self._inference_latencies, dtype=np.float64)
            gap = np.array(self._replan_intervals, dtype=np.float64)
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        snap = {"queue_size": float(queue_size), "replans": float(len(lat))}
        if len(lat) > 0:
            snap.update(
                infer_ms_p50=float(np.percentile(lat, 50) * 1e3),
                infer_ms_p95=float(np.percentile(lat, 95) * 1e3),
                infer_ms_max=float(lat.max() * 1e3),
            )
        if len(gap) > 0:
            snap["replan_interval_ms_p50"] = float(np.percentile(gap, 50) * 1e3)
        return snap
