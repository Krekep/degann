import subprocess
import sys
import shared


def main():
    shared.configure_python_path()
    subprocess.check_call(
        [sys.executable, "-m", "pytest", "-vv", "-s", str(shared.TESTS)]
    )


if __name__ == "__main__":
    main()
