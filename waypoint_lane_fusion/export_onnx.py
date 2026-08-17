#!/usr/bin/env python3
import argparse, torch
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("--output",default="artifacts/lane_resnet18.onnx"); a=p.parse_args()
model=torch.jit.load(a.model,map_location="cpu").eval(); Path(a.output).parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(model,torch.randn(1,3,224,224),a.output,input_names=["images"],output_names=["waypoint"],opset_version=13,dynamic_axes=None)
print(a.output)
