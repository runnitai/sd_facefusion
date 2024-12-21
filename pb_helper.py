import os

import tf2onnx
import tensorflow as tf

from facefusion.hash_helper import create_file_hash


def convert_pb_to_onnx(pb_path, onnx_path, input_names, output_names):
    """
    Converts a .pb TensorFlow model to ONNX format.
    Args:
        pb_path: Path to the .pb model file.
        onnx_path: Path to save the converted ONNX model.
        input_names: List of input tensor names.
        output_names: List of output tensor names.
    """
    with tf.Graph().as_default():
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        with tf.compat.v1.Session() as sess:
            tf_rep = tf2onnx.convert.from_graph_def(
                graph_def,
                input_names=input_names,
                output_names=output_names,
                opset=13  # Specify ONNX opset version
            )
            model_proto, _ = tf_rep
            with open(onnx_path, "wb") as f:
                f.write(model_proto.SerializeToString())


def convert_style_models():
    style_model_path = "E:\\dev\\stable-diffusion-webui\\models\\facefusion\\style\\"
    for file in os.listdir(style_model_path):
        out_file = os.path.join(style_model_path, file.replace(".pb", ".onnx"))
        if os.path.exists(out_file):
            os.remove(out_file)
        if file.endswith(".pb"):
            convert_pb_to_onnx(
                pb_path=os.path.join(style_model_path, file),
                onnx_path=out_file,
                input_names=["input_image:0"],
                output_names=["output_image:0"]
            )
            print(f"Converted {file} to {out_file}")
            if os.path.exists(out_file):
                create_file_hash(out_file)
