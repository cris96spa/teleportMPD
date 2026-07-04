import os
import subprocess
import sys


def main() -> int:
    """Main entry point for post-checkout hook.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    from_ref = os.getenv("PRE_COMMIT_FROM_REF")
    to_ref = os.getenv("PRE_COMMIT_TO_REF")
    checkout_type = os.getenv("PRE_COMMIT_CHECKOUT_TYPE")

    if from_ref == to_ref:
        return 0  # Same ref, nothing to do

    if from_ref is None:
        print("PRE_COMMIT_FROM_REF is None, aborting dependency check.")
        return 1
    if to_ref is None:
        print("PRE_COMMIT_TO_REF is None, aborting dependency check.")
        return 1

    # Only check on branch checkout (flag = 1)
    if checkout_type != "1":
        return 0

    if is_file_changed("pyproject.toml", from_ref, to_ref):
        print_warning()

    return 0


def is_file_changed(filename: str, prev_ref: str, new_ref: str) -> bool:
    """Check if a file changed between two git references."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", prev_ref, new_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        changed_files = result.stdout.strip().split("\n")
        return filename in changed_files
    except subprocess.CalledProcessError:
        return False


def print_warning() -> None:
    """Print warning that pyproject.toml has changed."""
    print("\n" + "=" * 70)
    print("⚠️  WARNING: pyproject.toml has changed!")
    print("=" * 70)
    print("Dependencies may be out of sync.")
    print("Please run `make dev` to ensure all dependencies are updated.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    sys.exit(main())
