import subprocess
import shared


def main():
    shared.configure_python_path()
    subprocess.check_call(["python", "-m", "pytest", "-vv", "-s", shared.TESTS])


if __name__ == "__main__":
    main()
