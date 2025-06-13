import sys
import ast
import onnx
import json
import argparse
import data_config
from pathlib import Path
from onnx import shape_inference, helper, TensorProto, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import lstm_decompose


def _get_args_parser():
    parser = argparse.ArgumentParser("Decompose LSTM nodes in default ONNX model.", add_help=False)
    parser.add_argument("--input-path", default="", help="Input .pth model path")
    parser.add_argument("--input-shape", type=ast.literal_eval, default="(1,3,224,224)",
                         help="Model input shape, default: (1,3,224,224)")
    parser.add_argument("--decompose-path",default="", help="Output ONNX model path")
    parser.add_argument("--batch-size", default=256, type=int, help="Batchsize used in accuracy tests")
    return parser


def _fill_default_args(args):
    default_input_path = Path("/home/yuxinr/far/FAR/hw_files/models/model.onnx")
    default_decomposed_path = Path(args.base_dir) / "hw_files" / "models" / "model_decomposed.onnx"
    if args.input_path == "":
        args.input_path = default_input_path
        print(f"Using default input .pth path {args.input_path}")
    if args.decompose_path == "":
        args.decompose_path = default_decomposed_path
        print(f"Using default output ONNX path {args.decompose_path}")
    
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LSTM nodes decompose.", 
                                     parents=[data_config.get_common_parser(), _get_args_parser()])
    args = parser.parse_args()
    args = _fill_default_args(args)
    args = data_config.fill_default_common_args(args)
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=4))

    # load dataset
    data_loader_val = data_config.load_onnx_val_dataset(args)

    # load ONNX model
    print("Loading input model...")
    original_model = onnx.load(args.input_path)
    print("Checking input model...")
    onnx.checker.check_model(original_model)
    print("Evaluating input model...")
    data_config.evaluate_onnx_model(args.input_path, data_loader_val)
    # print("IR version:", original_model.ir_version)
    # decompose model
    print("Decomposing input model...")
    decomposed_model = lstm_decompose.decompose_lstm(original_model)
    # print("IR version:", decomposed_model.ir_version)
    print("Checking decomposed model...")
    onnx.checker.check_model(decomposed_model)
    print("Saving decomposed model...")
    decomposed_model.ir_version = 10
    onnx.save(decomposed_model, args.decompose_path)
    print("Evaluating decomposed model...")
    data_config.evaluate_onnx_model(args.decompose_path, data_loader_val)

    # Patch empty loop cond
    print("Patching decomposed model...")
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["_const_true_for_loop_cond"],
        value=helper.make_tensor(
            name="_const_true_for_loop_cond" + "_lit",
            data_type=TensorProto.BOOL,
            dims=[],  # scalar
            vals=[True],
        ),
        name="_const_true_for_loop_cond" + "_node",
    )
    decomposed_model.graph.node.insert(0, const_node)
    patched = 0
    for node in decomposed_model.graph.node:
        if node.op_type != "Loop":
            continue
        if len(node.input) >= 2 and node.input[1] == "":
            node.input[1] = "_const_true_for_loop_cond"
            patched += 1
    print(f"{patched} nodes with empty loop condition fixed")
    inferred_model = SymbolicShapeInference.infer_shapes(decomposed_model, verbose=2)
