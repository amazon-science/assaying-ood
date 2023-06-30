import os
import os.path
import subprocess
from typing import List

import pytest
from PIL import Image


class CLIRunner:
    def __call__(
        self,
        arguments: List[str],
        expected_outputs: List[str] = None,
        expected_stderrs: List[str] = None,
        expect_successful: bool = True,
    ) -> None:
        result = subprocess.run(
            ["env", "PYTHONPATH=src", "python3", "bin/run.py"] + arguments,
            capture_output=True,
            text=True,
        )
        print(" ".join(["env", "PYTHONPATH=src", "python3", "bin/run.py"] + arguments))
        print("Stdout:")
        print(result.stdout)
        print("Stderr:")
        print(result.stderr)
        if expected_outputs:
            for expected_output in expected_outputs:
                assert expected_output in result.stdout, expected_output

        if expected_stderrs:
            for expected_stderr in expected_stderrs:
                assert expected_stderr in result.stderr, expected_stderr

        if expect_successful:
            if result.returncode:
                raise RuntimeError("Inspector terminated with non-zero return code.")
        else:
            if not result.returncode:
                # Run succeeds but we we expected it to fail.
                raise RuntimeError(
                    "Inspector terminated with zero return code, "
                    "yet a failure return code was expected."
                )


@pytest.fixture
def cli_runner() -> CLIRunner:
    return CLIRunner()


@pytest.fixture
def datadir(tmpdir, number_of_classes: int = 3, number_of_images: int = 2):
    for split in "train", "val", "test":
        splitdir = os.path.join(tmpdir, split)
        os.mkdir(splitdir)
        for class_id in range(number_of_classes):
            classdir = os.path.join(splitdir, f"class{class_id}")
            os.mkdir(classdir)

            for image_id in range(number_of_images):
                imagepath = os.path.join(classdir, f"image{image_id}.jpg")
                Image.new("RGB", (250, 250), color=(image_id, image_id, image_id)).save(imagepath)
    return tmpdir


@pytest.fixture
def s3dataset() -> str:
    return "S3TinyImageNetForTesting"


@pytest.fixture
def s3_fairness_dataset() -> str:
    return "S3FairFace-TinyTestInstance"


@pytest.fixture
def eval_target_attribute() -> str:
    return "gender"


@pytest.fixture
def eval_group_attribute() -> str:
    return "age"


@pytest.fixture
def adaptation_target_attribute() -> str:
    return "age"
