import os
import subprocess
import sys


def test_preprocess():

    result = subprocess.run(
        [
            sys.executable,
            "src/preprocess.py"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0

    assert os.path.exists(
        "artifacts/train.csv"
    )

    assert os.path.exists(
        "artifacts/test.csv"
    )