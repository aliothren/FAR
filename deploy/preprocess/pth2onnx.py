import sys
import ast
import json
import onnx
import torch
import argparse
import torch.onnx
from pathlib import Path
import data_config
# load modeling evaluation set for testing accuracy
modeling_dir = Path(__file__).resolve().parents[2] / "modeling"
sys.path.insert(0, str(modeling_dir))
from train import evaluate_model


def _get_args_parser():
    parser = argparse.ArgumentParser("Convert .pth model file to ONNX model", add_help=False)
    parser.add_argument("--pth-path", default="", help="Input .pth model path")
    parser.add_argument("--input-shape", type=ast.literal_eval, default="(1,3,224,224)",
                         help="Model input shape, default: (1,3,224,224)")
    parser.add_argument("--onnx-path",default="", help="Output ONNX model path")
    parser.add_argument("--batch-size", default=256, type=int, help="Batchsize used in accuracy tests")
    return parser


def _fill_default_args(args):
    default_pth_path = Path("/home/yuxinr/far/FAR/checkpoints/2025-06-03-14-45-48/model_seq0_parallel.pth")
    default_onnx_path = Path(args.base_dir) / "hw_files" / "models" / "model.onnx"
    if args.pth_path == "":
        args.pth_path = default_pth_path
        print(f"Using default input .pth path {args.pth_path}")
    if args.onnx_path == "":
        args.onnx_path = default_onnx_path
        print(f"Using default output ONNX path {args.onnx_path}")
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(".pth model to ONNX model", 
                                     parents=[data_config.get_common_parser(), _get_args_parser()])
    args = parser.parse_args()
    args = _fill_default_args(args)
    args = data_config.fill_default_common_args(args)
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=4))

    # load .pth model
    model = torch.load(args.pth_path)
    model.eval()

    # load dataset
    data_loader_val = data_config.load_onnx_val_dataset(args)

    # test .pth model accuracy
    model.to(args.device)
    test_stats = evaluate_model(data_loader_val, args.device, model)
    print(f"Accuracy on .pth model: {test_stats['acc1']:.1f}%")

    # convert to ONNX model
    model.to("cpu")
    dummy_input = torch.randn(args.input_shape)
    args.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        # verbose=True
    )
    print(f"\n ONNX model saved to: {args.onnx_path}")
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)

    # # check ONNX model nodes
    # print("\n=== ONNX model nodes list ===")
    # for i, node in enumerate(onnx_model.graph.node):
    #     print(f"{i:02d}: {node.op_type}  -> {node.name if node.name else '[NoName]'}")

    # test ONNX model accuracy
    data_config.evaluate_onnx_model(args.onnx_path, data_loader_val)
