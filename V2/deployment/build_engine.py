#!/usr/bin/env python3
"""Build a static FP16 TensorRT 7.x engine on the target Jetson."""
import argparse
from pathlib import Path
import tensorrt as trt

def build(onnx_path, engine_path, workspace_mb=512):
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors=[]
        for i in range(parser.num_errors): errors.append(str(parser.get_error(i)))
        raise RuntimeError('ONNX parse failed:\n'+'\n'.join(errors))
    if hasattr(builder,'create_builder_config'):
        config=builder.create_builder_config(); config.max_workspace_size=int(workspace_mb)*1024*1024
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        engine=builder.build_engine(network,config)
    else:
        builder.max_workspace_size=int(workspace_mb)*1024*1024; builder.fp16_mode=bool(builder.platform_has_fast_fp16)
        engine=builder.build_cuda_engine(network)
    if engine is None: raise RuntimeError('TensorRT returned no engine')
    Path(engine_path).parent.mkdir(parents=True,exist_ok=True)
    with Path(engine_path).open('wb') as stream: stream.write(engine.serialize())
    print('engine:',Path(engine_path).resolve(),'bytes:',Path(engine_path).stat().st_size,'fp16:',bool(builder.platform_has_fast_fp16))
    return engine

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--onnx',default='V2/models/waypoint_baseline.onnx'); p.add_argument('--engine',default='V2/models/waypoint_baseline_fp16.engine'); p.add_argument('--workspace-mb',type=int,default=512); a=p.parse_args(); build(a.onnx,a.engine,a.workspace_mb)

