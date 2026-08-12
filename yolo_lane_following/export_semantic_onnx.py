#!/usr/bin/env python3
"""Export the trained semantic checkpoint to a static Nano-friendly ONNX graph."""
import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper, numpy_helper
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def replace_static_splits(model: onnx.ModelProto) -> int:
    """Rewrite uneven constant-sized Split nodes for TensorRT 8.x.

    TensorRT on JetPack 4 can import an opset-13 Split incorrectly when its
    split sizes are supplied as a Constant input.  YOLO26's attention block
    contains a [32, 32, 64] split.  Equivalent static Slice nodes avoid the
    parser bug while preserving the ONNX graph semantics.
    """
    initializers = {x.name: x for x in model.graph.initializer}
    constants = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value = next((helper.get_attribute_value(a) for a in node.attribute
                      if a.name == "value"), None)
        if value is not None:
            constants[node.output[0]] = value

    rewritten = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Split":
            rewritten.append(node)
            continue
        split = next((helper.get_attribute_value(a) for a in node.attribute
                      if a.name == "split"), None)
        if split is None and len(node.input) == 2:
            value = initializers.get(node.input[1], constants.get(node.input[1]))
            if value is not None:
                split = numpy_helper.to_array(value).reshape(-1).tolist()
        if split is None or len(split) != len(node.output) or len(set(split)) == 1:
            rewritten.append(node)
            continue

        axis = next((helper.get_attribute_value(a) for a in node.attribute
                     if a.name == "axis"), 0)
        start = 0
        for index, (output, size) in enumerate(zip(node.output, split)):
            end = start + int(size)
            prefix = f"{node.name or 'split'}_{index}"
            names = [f"{prefix}_{suffix}" for suffix in ("starts", "ends", "axes", "steps")]
            for name, value in zip(names, (start, end, axis, 1)):
                model.graph.initializer.append(helper.make_tensor(
                    name, onnx.TensorProto.INT64, [1], [value]))
            rewritten.append(helper.make_node(
                "Slice", [node.input[0], *names], [output],
                name=f"{prefix}_Slice"))
            start = end
        replaced += 1

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return replaced


def replace_uint8_casts(model: onnx.ModelProto) -> int:
    """Use INT32 for class-map casts because TensorRT 8.x rejects UINT8."""
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == onnx.TensorProto.UINT8:
                attribute.i = onnx.TensorProto.INT32
                for value_info in (*model.graph.value_info, *model.graph.output):
                    if value_info.name in node.output:
                        value_info.type.tensor_type.elem_type = onnx.TensorProto.INT32
                replaced += 1
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=ROOT / "artifacts" / "track_yolo26n_sem_cube_best.pt")
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()
    exported = YOLO(str(args.model), task="semantic").export(
        format="onnx", imgsz=args.imgsz, batch=1, dynamic=False,
        simplify=False, opset=13, device="cpu")
    output = Path(exported)
    model = onnx.load(str(output))
    split_replaced = replace_static_splits(model)
    cast_replaced = replace_uint8_casts(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output))
    print(f"Semantic ONNX: {output.resolve()}")
    print(f"Replaced uneven static Split nodes: {split_replaced}; "
          f"UINT8 casts: {cast_replaced}; "
          f"operator types: {len(Counter(n.op_type for n in model.graph.node))}")


if __name__ == "__main__":
    main()
