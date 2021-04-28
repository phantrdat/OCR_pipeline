import argparse
import tensorrt as trt
import os


def build_engine(args, onnx_file_path, engine_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # builder params
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = args.batch_size
        builder.fp16_mode = builder.platform_has_fast_fp16
        # builder.int8_mode = builder.platform_has_fast_int8

        config = None
        # optimization profile params
        # config = builder.create_builder_config()
        # profile0 = builder.create_optimization_profile()
        # profile0.set_shape("input", [1, 3, 736, 1280], [1, 3, 736, 1280], [1, 3, 736, 1280])
        # config.add_optimization_profile(profile0)

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found.'.format(
                onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        # # Reshape input
        # network.get_input(0).shape = INPUT_SIZE
        print(f"Input shape: {network.get_input(0).shape}")
        print(f"output_y shape: {network.get_output(0).shape}, output_feature shape: {network.get_output(1).shape}")
        print('Completed parsing of ONNX file')
        print(f'Building an engine from file {onnx_file_path}; this may take a while...')
        if config is not None:
            engine = builder.build_engine(network, config)
        else:
            engine = builder.build_cuda_engine(network)
        print("Completed creating engine, writing to output file...")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--type', default='craft', type=str)
    parser.add_argument(
        '--craft_onnx', default='craft_text_detector/weights/craft_mlt_25k_da.onnx', type=str)
    parser.add_argument(
        '--arc_onnx', default='../weights/onnx/arc/ir50_asia-l2norm-b1.onnx', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--craft_width', type=int, default=768)
    parser.add_argument('--craft_height', type=int, default=768)
    parser.add_argument('--arc_width', type=int, default=112)
    parser.add_argument('--arc_height', type=int, default=112)
    args = parser.parse_args()

    TRT_LOGGER = trt.Logger()
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    if args.type == 'arc':
        # INPUT_SIZE = [args.batch_size, 3, args.arc_height, args.arc_width]
        onnx_file_path = args.arc_onnx
    elif args.type == 'craft':
        # INPUT_SIZE = [args.batch_size, 3, args.craft_height, args.craft_width]
        onnx_file_path = args.craft_onnx
    else:
        raise Exception("NOT IMPLEMENTED")
    engine_file_path = onnx_file_path.split('/')[-1].replace('onnx', 'engine')
    build_engine(args, onnx_file_path, engine_file_path)
