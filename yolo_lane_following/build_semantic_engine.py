#!/usr/bin/env python3
"""Build and verify the semantic TensorRT engine for the Python runtime."""

import argparse
from pathlib import Path

import tensorrt as trt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--workspace", type=int, default=512)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)

    if args.verify_only:
        payload = args.engine.read_bytes()
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(payload)
        if engine is None:
            raise SystemExit("TensorRT engine deserialization failed")
        print(f"Verified TensorRT {trt.__version__} engine: {args.engine}")
        return

    if not args.onnx.exists():
        raise SystemExit(f"Missing ONNX: {args.onnx}")
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise SystemExit("TensorRT ONNX parse failed")

    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace << 20
    config.set_flag(trt.BuilderFlag.FP16)
    print(f"Building TensorRT {trt.__version__} engine ({args.workspace} MiB workspace)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT engine build failed")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise SystemExit("TensorRT engine self-deserialization failed")
    print(f"Built and verified TensorRT {trt.__version__} engine: {args.engine}")
    print(f"Engine bytes: {args.engine.stat().st_size}")


if __name__ == "__main__":
    main()
