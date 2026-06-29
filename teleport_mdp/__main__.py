import argparse
import importlib
import pkgutil
import sys

from teleport_mdp import commands


def main():
    """Entry point for the teleport_mdp command-line interface (CLI).

    This module sets up the CLI, allowing users to execute various commands,
    defined in the `teleport_mdp.commands` package.

    Usage:
        1. Define the desired command and its arguments. You can use the `utils.commands` wrapper
           to create new commands easily.
        2. Run the CLI with the specified command and arguments.
    """
    parser = argparse.ArgumentParser(description="teleport_mdp CLI")

    # possible commands: file name inside /commands
    available_commands = []
    for submodule in pkgutil.iter_modules(commands.__path__):
        available_commands.append(submodule.name)
    parser.add_argument("command", choices=available_commands, help="Command to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the command")

    parsed_args = parser.parse_args()

    command = importlib.import_module(f"{commands.__package__}.{parsed_args.command}")
    if not hasattr(command, "main") or not callable(getattr(command, "main")):
        parser.error(f"Command '{parsed_args.command}' does not implement main()")

    # Remove the command name from sys.argv so that the command's argument parser works correctly
    sys.argv.pop(1)
    command.main()


if __name__ == "__main__":
    main()
