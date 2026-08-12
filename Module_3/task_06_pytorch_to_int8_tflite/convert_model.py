import os
import sys
import logging

import numpy as np

import torch

import onnx
from onnx2tf import convert

import tensorflow as tf

from model_definition import SimpleCNN

logging.basicConfig(
    filename="conversion_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

logger = logging.getLogger()

def load_model():

    if not os.path.exists("model.pth"):
        raise FileNotFoundError(
            "model.pth missing"
        )

    model = SimpleCNN()

    model.load_state_dict(
        torch.load(
            "model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    logger.info("Loaded model")

    return model


def validate_calibration():

    if not os.path.isdir("calib"):
        raise RuntimeError(
            "calib folder missing"
        )

    files = sorted(
        [
            f for f in os.listdir("calib")
            if f.endswith(".npy")
        ]
    )

    if len(files) == 0:
        raise RuntimeError(
            "No calibration files"
        )

    samples = []

    for file in files:

        path = os.path.join(
            "calib",
            file
        )

        arr = np.load(path)

        if arr.shape != (1, 28, 28):
            raise ValueError(
                f"Bad shape {arr.shape}"
            )

        if np.isnan(arr).any():
            raise ValueError(
                f"NaN in {file}"
            )

        if np.isinf(arr).any():
            raise ValueError(
                f"Inf in {file}"
            )

        arr = arr.astype(
            np.float32
        )

        samples.append(arr)

    logger.info(
        f"Validated {len(samples)} samples"
    )

    return samples


def export_onnx(model):

    dummy = torch.randn(
        1,
        1,
        28,
        28
    )

    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes=None
    )

    onnx_model = onnx.load(
        "model.onnx"
    )

    onnx.checker.check_model(
        onnx_model
    )

    logger.info(
        "ONNX validation passed"
    )


def convert_saved_model():

    convert(
        input_onnx_file_path="model.onnx",
        output_folder_path="saved_model"
    )

    logger.info(
        "SavedModel created"
    )



def representative_dataset(
    calibration_samples
):

    for sample in calibration_samples:

        sample = np.expand_dims(
            sample,
            axis=0
        )

        sample = np.transpose(
            sample,
            (0, 2, 3, 1)
        )

        yield [sample]


def convert_int8(calibration_samples):

    converter = tf.lite.TFLiteConverter.from_saved_model(
        "saved_model"
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        lambda:
        representative_dataset(
            calibration_samples
        )
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(
        "model_int8.tflite",
        "wb"
    ) as f:

        f.write(
            tflite_model
        )

    logger.info(
        "INT8 TFLite generated"
    )



def verify_tflite():

    interpreter = tf.lite.Interpreter(
        model_path="model_int8.tflite"
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    print(
        "Input dtype:",
        input_details[0]["dtype"]
    )

    print(
        "Output dtype:",
        output_details[0]["dtype"]
    )

    print(
        "Input quantization:",
        input_details[0]["quantization"]
    )

    print(
        "Output quantization:",
        output_details[0]["quantization"]
    )

    file_size = os.path.getsize(
        "model_int8.tflite"
    )

    print(
        "Model size:",
        file_size / 1024,
        "KiB"
    )

    dummy = np.zeros(
        input_details[0]["shape"],
        dtype=np.int8
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        dummy
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    print(
        "Inference output:"
    )

    print(output)


def main():

    model = load_model()

    calibration_samples = (
        validate_calibration()
    )

    export_onnx(model)

    convert_saved_model()

    convert_int8(
        calibration_samples
    )

    verify_tflite()

    logger.info(
        "Conversion completed"
    )


if __name__ == "__main__":
    main()