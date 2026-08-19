#!/usr/bin/env python3
"""Build and verify the semantic TensorRT engine for the Python runtime."""

import argparse
import os
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
        if not args.engine.exists():
            raise SystemExit(f"Missing engine: {args.engine}")
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
    # build_serialized_network was added after the TensorRT version commonly
    # shipped with JetPack 4. Keep this helper usable in the existing image.
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized = engine.serialize() if engine is not None else None
    if serialized is None:
        raise SystemExit("TensorRT engine build failed")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    # Do not destroy the last known-good engine if build/deserialization fails.
    temporary = args.engine.with_name(f".{args.engine.name}.tmp.{os.getpid()}")
    temporary.write_bytes(serialized)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(temporary.read_bytes())
    if engine is None:
        raise SystemExit("TensorRT engine self-deserialization failed")
    print(f"Built and verified TensorRT {trt.__version__} engine: {args.engine}")
    temporary.replace(args.engine)
    print(f"Engine bytes: {args.engine.stat().st_size}")


if __name__ == "__main__":
    main()
