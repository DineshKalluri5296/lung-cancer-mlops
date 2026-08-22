import subprocess
import sys


def run_script(script):

    print("\n" + "=" * 60)
    print(f"RUNNING: {script}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script],
        check=True
    )

    return result.returncode


def main():

    print("=" * 60)
    print("LUNG CANCER MLOPS PIPELINE")
    print("=" * 60)

    # Step 1
    run_script("src/preprocess.py")

    # Step 2
    run_script("src/train.py")

    # Step 3
    run_script("src/evaluate.py")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()