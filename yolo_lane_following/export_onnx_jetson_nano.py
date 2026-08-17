#!/usr/bin/env python3
"""Export a static ONNX graph compatible with JetPack 4.x TensorRT."""

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parent
# TensorRT 8.2 ONNX parser explicitly marks these operators unsupported.
FORBIDDEN_TRT82_OPS = {
    "BitShift", "Compress", "ConvInteger", "Det", "DynamicQuantizeLinear",
    "Hardmax", "IsInf", "MatMulInteger", "MeanVarianceNormalization", "Mod",
    "Multinomial", "NegativeLogLikelihoodLoss", "NonZero", "OneHot",
    "QLinearConv", "QLinearMatMul", "Reciprocal", "RoiAlign", "SequenceAt",
    "SequenceConstruct", "SequenceEmpty", "SequenceErase", "SequenceInsert",
    "SequenceLength", "Shrink", "SoftmaxCrossEntropyLoss", "StringNormalizer",
    "TfIdfVectorizer", "Unique", "Xor",
}


def replace_integer_mod(model: onnx.ModelProto) -> int:
    """Replace non-negative index ``a % b`` with ``a - (a / b) * b``.

    YOLO26 uses this single Mod only to map non-negative TopK indices to a class
    index. Integer ONNX Div truncates toward zero, so the replacement is exact.
    """
    rewritten = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Mod":
            rewritten.append(node)
            continue
        if any(attr.name == "fmod" and helper.get_attribute_value(attr) != 0
               for attr in node.attribute):
            raise RuntimeError("Cannot safely rewrite floating-point Mod")
        quotient = f"{node.output[0]}_quotient"
        product = f"{node.output[0]}_product"
        rewritten.extend([
            helper.make_node("Div", [node.input[0], node.input[1]], [quotient],
                             name=f"{node.name}_Div"),
            helper.make_node("Mul", [quotient, node.input[1]], [product],
                             name=f"{node.name}_Mul"),
            helper.make_node("Sub", [node.input[0], product], list(node.output),
                             name=f"{node.name}_Sub"),
        ])
        replaced += 1
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return replaced


def replace_static_splits(model: onnx.ModelProto) -> int:
    """Replace unequal static Split nodes with static Slice nodes.

    TensorRT 7.1 (JetPack 4.5.1) incorrectly imports an opset-13 ``Split``
    whose sizes are supplied by a Constant input: it assigns the first split
    size to every output. YOLO26's attention block uses exactly that pattern.
    ``Slice`` with constant bounds has the same semantics and is imported
    correctly by the JetPack 4.x parser.
    """
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    constants = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value = next((helper.get_attribute_value(attr) for attr in node.attribute
                      if attr.name == "value"), None)
        if value is not None:
            constants[node.output[0]] = value

    rewritten = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Split":
            rewritten.append(node)
            continue

        split = next((helper.get_attribute_value(attr) for attr in node.attribute
                      if attr.name == "split"), None)
        if split is None and len(node.input) == 2:
            value = initializers.get(node.input[1])
            if value is None:
                value = constants.get(node.input[1])
            if value is not None:
                split = numpy_helper.to_array(value).reshape(-1).tolist()
        if split is None or len(split) != len(node.output) or len(set(split)) == 1:
            rewritten.append(node)
            continue

        axis = next((helper.get_attribute_value(attr) for attr in node.attribute
                     if attr.name == "axis"), 0)
        start = 0
        for index, (output, size) in enumerate(zip(node.output, split)):
            end = start + int(size)
            prefix = f"{node.name or 'split'}_{index}"
            starts = f"{prefix}_starts"
            ends = f"{prefix}_ends"
            axes = f"{prefix}_axes"
            steps = f"{prefix}_steps"
            for name, value in ((starts, start), (ends, end), (axes, axis), (steps, 1)):
                model.graph.initializer.append(helper.make_tensor(
                    name, onnx.TensorProto.INT64, [1], [value]))
            rewritten.append(helper.make_node(
                "Slice", [node.input[0], starts, ends, axes, steps], [output],
                name=f"{prefix}_Slice"))
            start = end
        replaced += 1

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=ROOT / "artifacts" / "lane_yolo26n_seg_best.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "lane_yolo26n_seg_nano.onnx")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--skip-export", action="store_true",
                        help="patch the ONNX adjacent to --model instead of exporting again")
    args = parser.parse_args()

    raw = args.model.with_suffix(".onnx")
    if not args.skip_export:
        from ultralytics import YOLO

        exported = YOLO(str(args.model), task="segment").export(
            format="onnx", imgsz=args.imgsz, batch=1, opset=13, dynamic=False,
            simplify=False, nms=False, device="cpu")
        raw = Path(exported)
    if not raw.exists():
        raise SystemExit(f"Missing raw ONNX: {raw}")

    model = onnx.load(str(raw))
    replaced = replace_integer_mod(model)
    if replaced != 1:
        raise RuntimeError(f"Expected one YOLO26 index Mod node, found {replaced}")
    split_replaced = replace_static_splits(model)
    onnx.checker.check_model(model, full_check=True)
    ops = Counter(node.op_type for node in model.graph.node)
    unsupported = sorted(set(ops) & FORBIDDEN_TRT82_OPS)
    if unsupported:
        raise RuntimeError(f"JetPack 4.x TensorRT unsupported operators remain: {unsupported}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"Jetson Nano ONNX: {args.output.resolve()}")
    print("Input: batch=1, RGB FP32, NCHW 1x3x224x224 | opset=13 | static shapes")
    print(f"Replaced Mod nodes: {replaced}; uneven Split nodes: {split_replaced} | "
          f"operators: {len(ops)} types")


if __name__ == "__main__":
    main()
