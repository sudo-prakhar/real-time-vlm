"""Command-line entry point:  python -m gavi <command> [options]"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "monitor": ("gavi.monitor", "phase 1: stateless rule monitor — alert when a plain-English rule holds"),
    "world": ("gavi.world_monitor", "phase 2: world model — persistent tracking, event timeline, Q&A"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python -m gavi <command> [options]\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<9} {desc}")
        print("\nRun `python -m gavi <command> --help` for a command's options.")
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    module, _ = COMMANDS[sys.argv[1]]
    importlib.import_module(module).main(sys.argv[2:])


if __name__ == "__main__":
    main()
