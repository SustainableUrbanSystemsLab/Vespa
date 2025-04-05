import onnx
from onnx import TensorProto

# Load the ONNX model
model_path = "gbr_best_Y1_compat.onnx"
model = onnx.load(model_path)

# Get model graph
graph = model.graph

# ONNX data type mapping
onnx_dtype_map = {v: k for k, v in TensorProto.DataType.items()}

# Extract and print inputs
print("Model Inputs:")
for inp in graph.input:
    name = inp.name
    shape_dims = [dim.dim_value if dim.HasField("dim_value") else "dynamic" for dim in inp.type.tensor_type.shape.dim]
    dtype = onnx_dtype_map.get(inp.type.tensor_type.elem_type, "Unknown")
    
    print(f" Input Name: {name}")
    print(f"   Shape: {shape_dims}")
    print(f"   Data Type: {dtype}")
    print("-" * 50)

# Extract and print initializers (which might contain missing inputs)
print("\n Model Initializers (Possible Missing Inputs):")
for init in graph.initializer:
    print(f"Initializer Name: {init.name}, Shape: {init.dims}, Data Type: {onnx_dtype_map.get(init.data_type, 'Unknown')}")

# Check if any input is actually an initializer
initializer_names = {init.name for init in graph.initializer}
for inp in graph.input:
    if inp.name in initializer_names:
        print(f"⚠️ Warning: Input {inp.name} is actually an initializer (constant).")
print("\n Model Nodes:")
for node in graph.node:
    print(f"Node: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")
