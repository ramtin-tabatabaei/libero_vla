"""Send commands to a running robot_server.py.

    python robot_cli.py state
    python robot_cli.py move -0.12 -0.24 0.14
    python robot_cli.py moveto alphabet_soup_1
    python robot_cli.py grasp
    python robot_cli.py lift 0.2
    python robot_cli.py run example_waypoints.json
    python robot_cli.py pick milk_1  # hybrid mode: RRT grasp and hold
    python robot_cli.py vla "Put the milk in the basket." 250

With no arguments it opens an interactive prompt:

    python robot_cli.py
    robot> moveto alphabet_soup_1
    robot> grasp
"""

import socket
import sys

HOST = "127.0.0.1"
PORT = 5599


def send(text: str) -> str:
    try:
        with socket.create_connection((HOST, PORT), timeout=600) as s:
            s.sendall(text.encode())
            chunks = []
            while True:
                b = s.recv(65536)
                if not b:
                    break
                chunks.append(b)
            return b"".join(chunks).decode("utf-8", "replace")
    except ConnectionRefusedError:
        return (
            f"nothing listening on {HOST}:{PORT}\n"
            "start the sim first:  MUJOCO_GL=glx python robot_server.py"
        )
    except socket.timeout:
        return "timed out waiting for the sim"


def main():
    if len(sys.argv) > 1:
        print(send(" ".join(sys.argv[1:])))
        return

    print("interactive mode -- 'help' for commands, 'exit' to leave the prompt")
    while True:
        try:
            line = input("robot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("exit", "bye"):
            return
        print(send(line))


if __name__ == "__main__":
    main()
