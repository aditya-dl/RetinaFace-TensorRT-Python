import os
import argparse
import numpy as np
import tensorrt as trt
import common as common

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path, batch_size=1):
    """Builds a new TensorRT engine and serializes it."""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # create builder config object
        config = builder.create_builder_config()

        # The maximum workspace size. The maximum GPU temporary
        # memory which the engine can use at execution time.
        config.max_workspace_size = 1 << 30 # 256MiB

        # set batch size
        builder.max_batch_size = batch_size

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)

        print('Loading ONNX file from path {}...'.format(onnx_file_path))

        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        print("Network input shape: {}".format(network.get_input(0).shape))
        network.get_input(0).shape = [batch_size, 3, -1, -1]

        # Add optimisation profile
        profile = builder.create_optimization_profile()
        profile.set_shape("input0", (batch_size, 3, 32, 64), (batch_size, 3, 1080, 1920), (batch_size, 3, 2160, 3840)) 
        config.add_optimization_profile(profile)

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

        engine = builder.build_engine(network, config)

        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

def main(onnx_path=None, engine_path=None, batch_size=1):
    """Create a TensorRT engine for ONNX-based lpd and run inference."""

    # Try to load a previously generated RetinaFace mobilenet0.25 network graph in ONNX format:
    onnx_file_path = './FaceDetector.onnx' if onnx_path == None else onnx_path
    engine_file_path = './FaceDetector.engine' if engine_path == None else engine_path
    get_engine(onnx_file_path, engine_file_path, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-path', default=None, type=str, help='ONNX model file path')
    parser.add_argument('--engine-path', default=None, type=str, help='TensorRT engine output file path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    args = parser.parse_args()

    main(args.onnx_path, args.engine_path)
