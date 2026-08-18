import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep
)

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput
)

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

from sagemaker.inputs import TrainingInput


# -----------------------------
# Configuration
# -----------------------------

BUCKET = "lung-cancer-mlops-2026"

ROLE = "arn:aws:iam::821656895219:role/Project3-SageMakerRole"


# -----------------------------
# SageMaker session
# -----------------------------

session = sagemaker.Session()


# -----------------------------
# Preprocessing
# -----------------------------

processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE,
    instance_type="ml.t3.medium",
    instance_count=1
)

processor_args = processor.run(
    code="sagemaker_preprocess.py",

    inputs=[
        ProcessingInput(
            source=f"s3://{BUCKET}/data/survey_lung_cancer.csv",
            destination="/opt/ml/processing/input"
        )
    ],

    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train"
        ),

        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test"
        )
    ]
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    step_args=processor_args
)


# -----------------------------
# Training
# -----------------------------

estimator = SKLearn(
    entry_point="sagemaker_train.py",

    source_dir=".",

    role=ROLE,

    instance_type="ml.t3.medium",

    instance_count=1,

    framework_version="1.2-1",

    py_version="py3",

    sagemaker_session=session
)

training_args = estimator.fit(
    inputs={
        "train": TrainingInput(
            s3_data=
            preprocess_step
            .properties
            .ProcessingOutputConfig
            .Outputs["train"]
            .S3Output
            .S3Uri
        )
    }
)

train_step = TrainingStep(
    name="TrainModel",
    step_args=training_args
)


# -----------------------------
# Pipeline
# -----------------------------

pipeline = Pipeline(
    name="LungCancerMLOpsPipeline",

    steps=[
        preprocess_step,
        train_step
    ],

    sagemaker_session=session
)


# -----------------------------
# Create / update pipeline
# -----------------------------

pipeline.upsert(
    role_arn=ROLE
)

print("Pipeline created successfully!")

print("Pipeline name:")
print(pipeline.name)


# Start pipeline
execution = pipeline.start()

print("Pipeline execution started!")

print(execution.arn)