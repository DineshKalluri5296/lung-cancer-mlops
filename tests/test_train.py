import os
import subprocess
import sys


def test_train():

    # Make sure preprocessing has been completed
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

    # Run training
    result = subprocess.run(
        [
            sys.executable,
            "src/train.py"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0

    # Check model
    assert os.path.exists(
        "model/model.pkl"
    )