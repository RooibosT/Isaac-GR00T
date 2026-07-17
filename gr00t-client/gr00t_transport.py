"""Minimal standalone client transport for the Isaac-GR00T PolicyServer.

Speaks the same ZMQ REQ/REP + msgpack_numpy wire protocol as
``gr00t.policy.server_client`` WITHOUT importing the ``gr00t`` package, so it
can run in the lightweight desktop env (needs only: pyzmq, msgpack,
msgpack-numpy, numpy).
"""

import functools
import io
import json
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _encode(obj, chain=None):
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
        raise TypeError("Refusing to encode object-dtype ndarray (would invoke pickle).")
    return mnp.encode(obj, chain=chain)


def _decode(obj, chain=None):
    if isinstance(obj, dict):
        # Custom ndarray-class envelope used by the gr00t server.
        marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
        if marker:
            payload = obj.get("as_npy", obj.get(b"as_npy"))
            if payload is None:
                raise ValueError("Malformed ndarray payload: 'as_npy' missing")
            return np.load(io.BytesIO(payload), allow_pickle=False)
        # ModalityConfig envelope -> plain dict (we don't have the gr00t class here).
        if "__ModalityConfig__" in obj or b"__ModalityConfig__" in obj:
            payload = obj.get("as_json", obj.get(b"as_json"))
            if isinstance(payload, bytes):
                payload = payload.decode()
            if isinstance(payload, str):
                payload = json.loads(payload)
            return payload
        # Refuse pickle-bearing object-dtype ndarray payloads.
        nd_val = obj.get(b"nd", obj.get("nd"))
        kind_val = obj.get(b"kind", obj.get("kind"))
        if nd_val and kind_val in (b"O", "O"):
            raise ValueError("Refusing to decode object-dtype ndarray payload.")
    return mnp.decode(obj, chain=chain)


def to_bytes(data: Any) -> bytes:
    return msgpack.packb(data, default=functools.partial(_encode))


def from_bytes(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=functools.partial(_decode), raw=False)


class Gr00tRemoteClient:
    """ZMQ REQ client for the Isaac-GR00T PolicyServer endpoints."""

    def __init__(self, host: str, port: int, timeout_ms: int = 15000, api_token: str | None = None):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data if data is not None else {}
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # REQ socket is stuck waiting for a reply that will never arrive;
            # recreate it so the next call can send again.
            old = self.socket
            self._init_socket()
            old.close(linger=0)
            raise
        response = from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, zmq.error.Again):
            return False

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": None})
        # Server returns (action_dict, info) — msgpack turns the tuple into a list.
        action, _info = response
        return action

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def reset(self) -> None:
        self.call_endpoint("reset", {"options": None})

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
            self.context.term()
        except Exception:
            pass
