"""Export helper; it exits honestly when the local ONNX stack cannot export."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); a=p.parse_args()
try:
 import torch
 model=torch.jit.load(a.checkpoint,map_location='cpu').eval(); dummy=torch.zeros(1,3,224,224)
 torch.onnx.export(model,dummy,a.output,opset_version=13,input_names=['images'],output_names=['mask_logits'],do_constant_folding=True)
 print('ONNX:',Path(a.output).resolve())
except Exception as e:
 raise SystemExit('ONNX export unavailable or failed: %s' % e)

