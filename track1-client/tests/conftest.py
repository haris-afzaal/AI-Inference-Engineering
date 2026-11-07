import pathlib
import pytest
import subprocess
import sys
import time
import socket


SERVER_SCRIPT = pathlib.Path(__file__).parent / "fake_stream_server.py"


def wait_for_port(host: str, port: int, timeout: float = 10.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server {host}:{port} did not start within {timeout}s")


@pytest.fixture(scope="session", autouse=True)
def start_fake_stream_server():
    process = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        wait_for_port("127.0.0.1", 8765, timeout=10.0)
    except RuntimeError:
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        raise RuntimeError(
            f"fake_stream_server failed to start.\n"
            f"STDOUT: {stdout.decode(errors='replace')}\n"
            f"STDERR: {stderr.decode(errors='replace')}"
        )

    yield process

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
