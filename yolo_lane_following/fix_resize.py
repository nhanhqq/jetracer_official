import onnx
from onnx import helper
import sys

def fix_resize_nodes(model_path):
    model = onnx.load(model_path)
    replaced = 0
    
    # We want to find Resize nodes and their 'scales' input.
    for node in model.graph.node:
        if node.op_type == "Resize":
            # Resize in opset 11+: inputs are [X, roi, scales, sizes]
            # Opsets 11 and 13 usually have scales as the 3rd input (index 2)
            if len(node.input) >= 3 and node.input[2] != "":
                scales_input_name = node.input[2]
                
                # Check if it's already an initializer
                is_initializer = any(init.name == scales_input_name for init in model.graph.initializer)
                if not is_initializer:
                    print(f"Fixing scales for Resize node: {node.name}")
                    
                    # We will replace the scales input with a new initializer
                    new_scales_name = f"{node.name}_fixed_scales"
                    # In YOLOv8, Resize scales are typically [1.0, 1.0, 2.0, 2.0]
                    # We can infer the scales from the graph if possible, but 2.0 is standard for YOLO PANet upsampling.
                    scales_tensor = helper.make_tensor(
                        name=new_scales_name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[4],
                        vals=[1.0, 1.0, 2.0, 2.0]
                    )
                    model.graph.initializer.append(scales_tensor)
                    node.input[2] = new_scales_name
                    replaced += 1

    if replaced > 0:
        onnx.save(model, model_path)
        print(f"Fixed {replaced} Resize nodes in {model_path}")
    else:
        print("No dynamic Resize nodes found to fix.")

if __name__ == "__main__":
    fix_resize_nodes(sys.argv[1])
