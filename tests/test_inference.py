import os
import subprocess
import sys


def test_inference():

    # Make sure model exists
    if not os.path.exists(
        "model/model.pkl"
    ):

        if not os.path.exists(
            "artifacts/train.csv"
        ):

            subprocess.run(
                [
                    sys.executable,
                    "src/preprocess.py"
                ],
                check=True
            )

        subprocess.run(
            [
                sys.executable,
                "src/train.py"
            ],
            check=True
        )

    # Run inference
    result = subprocess.run(
        [
            sys.executable,
            "src/inference.py"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0

    # Check that prediction was produced
    assert (
        "Prediction" in result.stdout
        or "prediction" in result.stdout
    )