"""Interruptible client for the existing OpenPI websocket model server."""
import time
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


class Cancelled(Exception):
    pass


class OpenPiClient:
    def __init__(self, host, port, stop):
        self.stop = stop
        self._ws = connect(f"ws://{host}:{port}", compression=None, max_size=None,
                           open_timeout=5, close_timeout=1)
        try:
            self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=5))
        except BaseException:
            self._ws.close()
            raise

    def reset(self):
        pass

    def infer(self, observation):
        self._ws.send(msgpack_numpy.packb(observation))
        deadline = time.monotonic() + 900
        while not self.stop.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("The policy did not return an action within 15 minutes")
            try:
                response = self._ws.recv(timeout=0.2)
            except TimeoutError:
                continue
            if isinstance(response, str):
                raise RuntimeError(response)
            return msgpack_numpy.unpackb(response)
        raise Cancelled()
