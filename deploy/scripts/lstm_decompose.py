"""
Decompose the LSTM ONNX op.
Outputs an ONNX graph with the internal operations as separate nodes.

Example command line:
PYTHONPATH=. python3 quantize/transforms/lstm_decompose.py \
--input_model_path quantize/transforms/org_model.onnx \
--output_model_path quantize/transforms/new_model.onnx
"""

import argparse
import pathlib

import numpy as np
import onnx
from onnxscript import FLOAT, script
from onnxscript import opset14 as op

from common.rustwork_onnx import ModelNetwork, SubgraphTemplate

SEQ_LENGTH_STR = "seq_length"
BATCH_SIZE_STR = "batch_size"
INPUT_SIZE_STR = "input_size"
NUM_DIRECTIONS_STR = "num_directions"
HIDDEN_SIZE_STR = "hidden_size"


def _rename_subgraph(graph, suffix):
    """Renames all node names, inputs and outputs within a graph by adding a suffix"""
    for node in graph.node:
        if node.name:
            node.name += suffix
        if node.input:
            for input_index in range(len(node.input)):
                if node.input[input_index]:
                    node.input[input_index] += suffix
        if node.output:
            for output_index in range(len(node.output)):
                if node.output[output_index]:
                    node.output[output_index] += suffix
        if node.attribute:
            for attribute in node.attribute:
                if attribute.t and attribute.t.name:
                    attribute.t.name += suffix
                if attribute.g and attribute.g.node:
                    _rename_subgraph(attribute.g, suffix)
    for input in graph.input:
        if input.name:
            input.name += suffix
    for output in graph.output:
        if output.name:
            output.name += suffix


def _rename_subgraph_inputs(graph, old_name, new_name):
    """Renames graph inputs that match old_name, replacing them with new_name"""
    for node in graph.node:
        if node.input:
            for input_index in range(len(node.input)):
                if node.input[input_index] == old_name:
                    node.input[input_index] = new_name
        if node.attribute:
            for attribute in node.attribute:
                if attribute.g and attribute.g.node:
                    _rename_subgraph_inputs(attribute.g, old_name, new_name)


def _rename_subgraph_outputs(graph, old_name, new_name):
    """Renames graph outputs that match old_name, replacing them with new_name"""
    for node in graph.node:
        if node.output:
            for output_index in range(len(node.output)):
                if node.output[output_index] == old_name:
                    node.output[output_index] = new_name


def _add_cond_to_loop(graph):
    """Adds constant boolean cond to loop to allow for symbolic shape inference"""
    patched_num = 0
    for node in graph.node:
        if node.op_type == "Loop":
            node.input[1] = node.name + "_cond"
            graph.initializer.append(
                onnx.helper.make_tensor(
                    name=node.name + "_cond",
                    data_type=onnx.TensorProto.BOOL,
                    dims=[1],
                    vals=[True],
                )
            )
            patched_num += 1
    print(f"Patched {patched_num} empty loops condition as True")


def _patch_output_type(original_model, new_model):
    """Fix possible lack of output shape of decomposed model,
    by copy output parameters from the original model.
    Inputs: original (not decomposed) model, target (decomposed) model
    Output: decomposed model"""
    original_output_info = {out.name: out.type for out in original_model.graph.output}

    for out in new_model.graph.output:
        if out.name not in original_output_info:
            print(f"Warning: output {out.name} not found in original model.")
            continue
        if (
            not out.type.HasField("tensor_type")
            or not out.type.tensor_type.HasField("shape")
            or len(out.type.tensor_type.shape.dim) == 0
        ):
            out.type.CopyFrom(original_output_info[out.name])

    return new_model


def decompose_lstm(input_model):
    """Decompose all LSTM ONNX ops within model.
    Outputs an ONNX graph with the internal operations of LSTM as separate nodes."""
    model_network = ModelNetwork(input_model)

    # Create LSTM subgraph template for matching
    template = SubgraphTemplate()
    template.add_operator_of_type(op_type="LSTM")

    matched_graphs = list(model_network.find_subgraphs(template, disjoint=True))
    for i, match in enumerate(matched_graphs):
        lstm_node = match.get_node_data(0).onnx_node

        # Check LSTM op attributes and optional inputs are supported
        # TODO: Add support for other attributes and optional inputs
        required_model_attributes = {
            "activation_alpha": [None],
            "activation_beta": [None],
            "activations": [[b"Sigmoid", b"Tanh", b"Tanh"]],
            "clip": [None],
            "direction": [b"forward", b"bidirectional"],
            "input_forget": [0],
            "layout": [0],
        }
        for attr in lstm_node.attribute:
            if attr.name in required_model_attributes:
                assert (
                    onnx.helper.get_attribute_value(attr)
                    in required_model_attributes[attr.name]
                ), "Attribute {}: {} not supported".format(
                    attr.name, onnx.helper.get_attribute_value(attr)
                )
                if attr.name == "direction":
                    lstm_direction = onnx.helper.get_attribute_value(attr)
        assert lstm_node.input[3], "B (bias tensor) must be provided"
        assert not lstm_node.input[4], "sequence_lens not supported"
        assert lstm_node.input[5], "initial_h must be provided"
        assert lstm_node.input[6], "initial_c must be provided"
        assert (
            len(lstm_node.input) < 8 or not lstm_node.input[7]
        ), "P (weight tensor for peepholes) not supported"

        # Construct decomposed LSTM
        W = model_network.convert_initializer_to_np_array(lstm_node.input[1])
        R = model_network.convert_initializer_to_np_array(lstm_node.input[2])
        B = model_network.convert_initializer_to_np_array(lstm_node.input[3])
        if lstm_direction == b"forward":
            new_lstm = lstm_forward_model(W, R, B).to_model_proto()
        elif lstm_direction == b"bidirectional":
            new_lstm = lstm_bidirectional_model(W, R, B).to_model_proto()

        output_names = [output_info.name for output_info in new_lstm.graph.output]

        # Rename LSTM with suffix
        suffix = "_" + (lstm_node.name or str(i))
        _rename_subgraph(new_lstm.graph, suffix)

        # Change input and output names of new LSTM
        new_lstm.graph.input[0].name = lstm_node.input[0]
        _rename_subgraph_inputs(new_lstm.graph, "X" + suffix, lstm_node.input[0])
        new_lstm.graph.input[1].name = lstm_node.input[5]
        _rename_subgraph_inputs(
            new_lstm.graph, "initial_h" + suffix, lstm_node.input[5]
        )
        new_lstm.graph.input[2].name = lstm_node.input[6]
        _rename_subgraph_inputs(
            new_lstm.graph, "initial_c" + suffix, lstm_node.input[6]
        )
        new_lstm.graph.output[0].name = lstm_node.output[0]
        _rename_subgraph_outputs(
            new_lstm.graph, output_names[0] + suffix, lstm_node.output[0]
        )
        new_lstm.graph.output[1].name = lstm_node.output[1]
        _rename_subgraph_outputs(
            new_lstm.graph, output_names[1] + suffix, lstm_node.output[1]
        )
        new_lstm.graph.output[2].name = lstm_node.output[2]
        _rename_subgraph_outputs(
            new_lstm.graph, output_names[2] + suffix, lstm_node.output[2]
        )

        model_network.substitute_matched_graph(match, new_lstm)

    new_onnx_model = model_network.as_onnx_model_proto()
    # Add cond to loop
    _add_cond_to_loop(new_onnx_model.graph)
    # Add output type if missing
    new_onnx_model = _patch_output_type(input_model, new_onnx_model)
    onnx.checker.check_model(new_onnx_model)
    return new_onnx_model


def lstm_forward_model(W, R, B):
    """
    Constructs a decomposed forward LSTM. Takes the weight, recurrence and bias tensors
    and saves them as constants. Returns OnnxFunction with internal operations of
    LSTM as separate nodes.
    TODO: This only works for forward LSTM, add support for backward LSTM

    Parameters
    ----------
    W : numpy.ndarray
        Weight matrix of shape (1, 4 * hidden_size, input_size)

    R : numpy.ndarray
        Recurrent weight matrix of shape (1, 4 * hidden_size, hidden_size)

    B : numpy.ndarray
        Bias vector of shape (1, 8 * hidden_size)

    Returns
    -------
    lstm_forward : OnnxFunction
        A function that performs the LSTM forward pass. It takes two inputs:
        - X: Input sequence of shape (seq_length, batch_size, input_size).
        - initial_h: Initial hidden state of shape (1, batch_size, hidden_size).
        - initial_c: Initial value of cell of shape (1, batch_size, hidden_size).

        It returns:
        - Y: Output sequence of shape (seq_length, 1, batch_size, hidden_size).
        - Y_h: Final hidden state of shape (1, batch_size, hidden_size).
        - Y_c: Final cell state of shape (1, batch_size, hidden_size).
    """
    W_i, W_o, W_f, W_c = np.split(np.squeeze(W, axis=0), 4, axis=0)
    W_i_T_np, W_o_T_np, W_f_T_np, W_c_T_np = (
        np.transpose(W_i),
        np.transpose(W_o),
        np.transpose(W_f),
        np.transpose(W_c),
    )
    R_i, R_o, R_f, R_c = np.split(np.squeeze(R, axis=0), 4, axis=0)
    R_i_T_np, R_o_T_np, R_f_T_np, R_c_T_np = (
        np.transpose(R_i),
        np.transpose(R_o),
        np.transpose(R_f),
        np.transpose(R_c),
    )
    Wb_i_np, Wb_o_np, Wb_f_np, Wb_c_np, Rb_i_np, Rb_o_np, Rb_f_np, Rb_c_np = np.split(
        np.squeeze(B, axis=0), 8, axis=0
    )

    @script()
    def lstm(
        X: FLOAT[SEQ_LENGTH_STR, BATCH_SIZE_STR, INPUT_SIZE_STR],
        initial_h: FLOAT[1, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
        initial_c: FLOAT[1, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
    ) -> tuple[
        FLOAT[SEQ_LENGTH_STR, 1, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
        FLOAT[1, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
        FLOAT[1, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
    ]:
        WR_i_T = op.Constant(value=np.vstack((W_i_T_np, R_i_T_np)).astype(np.float32))
        WR_o_T = op.Constant(value=np.vstack((W_o_T_np, R_o_T_np)).astype(np.float32))
        WR_f_T = op.Constant(value=np.vstack((W_f_T_np, R_f_T_np)).astype(np.float32))
        WR_c_T = op.Constant(value=np.vstack((W_c_T_np, R_c_T_np)).astype(np.float32))
        WRb_i = op.Constant(value=(Wb_i_np + Rb_i_np).astype(np.float32))
        WRb_o = op.Constant(value=(Wb_o_np + Rb_o_np).astype(np.float32))
        WRb_f = op.Constant(value=(Wb_f_np + Rb_f_np).astype(np.float32))
        WRb_c = op.Constant(value=(Wb_c_np + Rb_c_np).astype(np.float32))

        shape_X = op.Shape(X)
        seq_length = op.Gather(shape_X, op.Constant(value=0))

        H_t = initial_h[0]
        C_t = initial_c[0]
        shape_H_T = op.Shape(op.Unsqueeze(H_t, axes=[0]))
        shape_index = op.Shape(op.Unsqueeze(op.Unsqueeze(H_t, axes=[0]), axes=[0]))
        shape_Y = op.Concat(op.Unsqueeze(seq_length, axes=[0]), shape_H_T, axis=0)
        Y = op.Expand(op.Unsqueeze(op.Unsqueeze(H_t, axes=[0]), axes=[0]), shape_Y)

        for t in range(seq_length):
            X_t = X[t]
            concat_X_H = op.Concat(X_t, H_t, axis=1)
            i_t = op.Sigmoid(op.Gemm(concat_X_H, WR_i_T, WRb_i))
            f_t = op.Sigmoid(op.Gemm(concat_X_H, WR_f_T, WRb_f))
            c_t = op.Tanh(op.Gemm(concat_X_H, WR_c_T, WRb_c))
            C_t = f_t * C_t + i_t * c_t
            o_t = op.Sigmoid(op.Gemm(concat_X_H, WR_o_T, WRb_o))
            H_t = o_t * op.Tanh(C_t)
            t_broadcast = op.Expand(
                op.Reshape(t, op.Constant(value=np.array([1], dtype=np.int64))),
                shape_index,
            )
            Y = op.ScatterElements(
                Y,
                t_broadcast,
                op.Unsqueeze(op.Unsqueeze(H_t, axes=[0]), axes=[0]),
                axis=0,
            )

        Y_h = op.Unsqueeze(H_t, axes=[0])
        Y_c = op.Unsqueeze(C_t, axes=[0])

        return Y, Y_h, Y_c

    return lstm


def lstm_bidirectional_model(W, R, B):
    """
    Constructs a decomposed bidirectional LSTM.
    Takes the weight, recurrence and bias tensors and saves them as constants.
    Returns OnnxFunction with internal operations of LSTM as separate nodes.
    Each bidirectional LSTM will be decomposed into 2 loops,
    for forward part and backward part separately.

    Parameters
    ----------
    W : numpy.ndarray
        Weight matrix of shape (2, 4 * hidden_size, input_size)

    R : numpy.ndarray
        Recurrent weight matrix of shape (2, 4 * hidden_size, hidden_size)

    B : numpy.ndarray
        Bias vector of shape (2, 8 * hidden_size)

    Returns
    -------
    lstm_bidirectional : OnnxFunction
        A function that performs the LSTM bidirectional pass. It takes two inputs:
        - X: Input sequence of shape (seq_length, batch_size, input_size).
        - initial_h: Initial hidden state of shape (2, batch_size, hidden_size).
        - initial_c: Initial value of cell of shape (2, batch_size, hidden_size).

        It returns:
        - Y: Output sequence of shape (seq_length, 2, batch_size, hidden_size).
        - Y_h: Final hidden state of shape (2, batch_size, hidden_size).
        - Y_c: Final cell state of shape (2, batch_size, hidden_size).
    """
    W, WB = np.split(W, 2, axis=0)  # [1, 4H, I]
    W_i, W_o, W_f, W_c = np.split(np.squeeze(W, axis=0), 4, axis=0)  # [H, I]
    WB_i, WB_o, WB_f, WB_c = np.split(np.squeeze(WB, axis=0), 4, axis=0)
    W_i_T_np, W_o_T_np, W_f_T_np, W_c_T_np = (  # [I, H]
        np.transpose(W_i),
        np.transpose(W_o),
        np.transpose(W_f),
        np.transpose(W_c),
    )
    WB_i_T_np, WB_o_T_np, WB_f_T_np, WB_c_T_np = (
        np.transpose(WB_i),
        np.transpose(WB_o),
        np.transpose(WB_f),
        np.transpose(WB_c),
    )

    R, RB = np.split(R, 2, axis=0)  # [1, 4H, H]
    R_i, R_o, R_f, R_c = np.split(np.squeeze(R, axis=0), 4, axis=0)  # [H, H]
    RB_i, RB_o, RB_f, RB_c = np.split(np.squeeze(RB, axis=0), 4, axis=0)
    R_i_T_np, R_o_T_np, R_f_T_np, R_c_T_np = (  # [H, H]
        np.transpose(R_i),
        np.transpose(R_o),
        np.transpose(R_f),
        np.transpose(R_c),
    )
    RB_i_T_np, RB_o_T_np, RB_f_T_np, RB_c_T_np = (
        np.transpose(RB_i),
        np.transpose(RB_o),
        np.transpose(RB_f),
        np.transpose(RB_c),
    )
    B, BB = np.split(B, 2, axis=0)  # [1, 8H]
    Wb_i_np, Wb_o_np, Wb_f_np, Wb_c_np, Rb_i_np, Rb_o_np, Rb_f_np, Rb_c_np = np.split(
        np.squeeze(B, axis=0), 8, axis=0
    )  # [H]
    WBb_i_np, WBb_o_np, WBb_f_np, WBb_c_np, RBb_i_np, RBb_o_np, RBb_f_np, RBb_c_np = (
        np.split(np.squeeze(BB, axis=0), 8, axis=0)
    )

    @script()
    def lstm_bidirectional(
        X: FLOAT[SEQ_LENGTH_STR, BATCH_SIZE_STR, INPUT_SIZE_STR],
        initial_h: FLOAT[2, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
        initial_c: FLOAT[2, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
    ) -> tuple[
        FLOAT[SEQ_LENGTH_STR, 2, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
        FLOAT[2, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
        FLOAT[2, BATCH_SIZE_STR, HIDDEN_SIZE_STR],
    ]:
        WR_i_T = op.Constant(value=np.vstack((W_i_T_np, R_i_T_np)).astype(np.float32))
        WR_o_T = op.Constant(value=np.vstack((W_o_T_np, R_o_T_np)).astype(np.float32))
        WR_f_T = op.Constant(value=np.vstack((W_f_T_np, R_f_T_np)).astype(np.float32))
        WR_c_T = op.Constant(value=np.vstack((W_c_T_np, R_c_T_np)).astype(np.float32))
        WRb_i = op.Constant(value=(Wb_i_np + Rb_i_np).astype(np.float32))
        WRb_o = op.Constant(value=(Wb_o_np + Rb_o_np).astype(np.float32))
        WRb_f = op.Constant(value=(Wb_f_np + Rb_f_np).astype(np.float32))
        WRb_c = op.Constant(value=(Wb_c_np + Rb_c_np).astype(np.float32))

        WRB_i_T = op.Constant(
            value=np.vstack((WB_i_T_np, RB_i_T_np)).astype(np.float32)
        )
        WRB_o_T = op.Constant(
            value=np.vstack((WB_o_T_np, RB_o_T_np)).astype(np.float32)
        )
        WRB_f_T = op.Constant(
            value=np.vstack((WB_f_T_np, RB_f_T_np)).astype(np.float32)
        )
        WRB_c_T = op.Constant(
            value=np.vstack((WB_c_T_np, RB_c_T_np)).astype(np.float32)
        )
        WRBb_i = op.Constant(value=(WBb_i_np + RBb_i_np).astype(np.float32))
        WRBb_o = op.Constant(value=(WBb_o_np + RBb_o_np).astype(np.float32))
        WRBb_f = op.Constant(value=(WBb_f_np + RBb_f_np).astype(np.float32))
        WRBb_c = op.Constant(value=(WBb_c_np + RBb_c_np).astype(np.float32))

        shape_X = op.Shape(X)
        seq_length = op.Gather(shape_X, op.Constant(value=0))

        H_t_forward, H_t_backward = initial_h[0], initial_h[1]
        C_t_forward, C_t_backward = initial_c[0], initial_c[1]
        shape_H_T = op.Shape(op.Unsqueeze(H_t_forward, axes=[0]))
        shape_index = op.Shape(
            op.Unsqueeze(op.Unsqueeze(H_t_forward, axes=[0]), axes=[0])
        )
        Y_shape = op.Concat(op.Unsqueeze(seq_length, axes=[0]), shape_H_T, axis=0)
        Y_forward = op.ConstantOfShape(Y_shape)
        Y_backward = op.ConstantOfShape(Y_shape)

        for t in range(seq_length):
            X_t_forward = X[t]  # [B, I]
            concat_X_H_forward = op.Concat(X_t_forward, H_t_forward, axis=1)  # [B, I+H]
            i_t_forward = op.Sigmoid(op.Gemm(concat_X_H_forward, WR_i_T, WRb_i))
            f_t_forward = op.Sigmoid(op.Gemm(concat_X_H_forward, WR_f_T, WRb_f))
            c_t_forward = op.Tanh(op.Gemm(concat_X_H_forward, WR_c_T, WRb_c))
            C_t_forward = f_t_forward * C_t_forward + i_t_forward * c_t_forward
            o_t_forward = op.Sigmoid(op.Gemm(concat_X_H_forward, WR_o_T, WRb_o))
            H_t_forward = o_t_forward * op.Tanh(C_t_forward)  # [B, H]
            t_broadcast_forward = op.Expand(
                op.Reshape(t, op.Constant(value=np.array([1], dtype=np.int64))),
                shape_index,
            )
            Y_forward = op.ScatterElements(
                Y_forward,
                t_broadcast_forward,
                op.Unsqueeze(op.Unsqueeze(H_t_forward, axes=[0]), axes=[0]),
                axis=0,
            )

        batch_size = op.Gather(op.Shape(X), op.Constant(value=1))
        seq_lens = op.Expand(
            op.Unsqueeze(seq_length, axes=[0]),
            op.Unsqueeze(batch_size, axes=[0]),
        )
        X_backward = op.ReverseSequence(X, seq_lens, batch_axis=1, time_axis=0)
        for t in range(seq_length):
            X_t_backward = X_backward[t]
            concat_X_H_backward = op.Concat(X_t_backward, H_t_backward, axis=1)
            i_t_backward = op.Sigmoid(op.Gemm(concat_X_H_backward, WRB_i_T, WRBb_i))
            f_t_backward = op.Sigmoid(op.Gemm(concat_X_H_backward, WRB_f_T, WRBb_f))
            c_t_backward = op.Tanh(op.Gemm(concat_X_H_backward, WRB_c_T, WRBb_c))
            C_t_backward = f_t_backward * C_t_backward + i_t_backward * c_t_backward
            o_t_backward = op.Sigmoid(op.Gemm(concat_X_H_backward, WRB_o_T, WRBb_o))
            H_t_backward = o_t_backward * op.Tanh(C_t_backward)
            t_backward = op.Sub(op.Sub(seq_length, t), op.Constant(value=1))
            t_broadcast_backward = op.Expand(
                op.Reshape(
                    t_backward, op.Constant(value=np.array([1], dtype=np.int64))
                ),
                shape_index,
            )
            Y_backward = op.ScatterElements(
                Y_backward,
                t_broadcast_backward,
                op.Unsqueeze(op.Unsqueeze(H_t_backward, axes=[0]), axes=[0]),
                axis=0,
            )

        Y = op.Concat(Y_forward, Y_backward, axis=1)
        Y_h = op.Concat(
            op.Unsqueeze(H_t_forward, axes=[0]),
            op.Unsqueeze(H_t_backward, axes=[0]),
            axis=0,
        )
        Y_c = op.Concat(
            op.Unsqueeze(C_t_forward, axes=[0]),
            op.Unsqueeze(C_t_backward, axes=[0]),
            axis=0,
        )

        return Y, Y_h, Y_c

    return lstm_bidirectional


def _get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose all LSTM ONNX ops within model"
    )
    parser.add_argument(
        "--input_model_path",
        type=pathlib.Path,
        help="Input ONNX model path",
        required=True,
    )
    parser.add_argument(
        "--output_model_path",
        type=pathlib.Path,
        help="Output ONNX model path",
        required=True,
    )
    return parser


def main():
    parser = _get_args_parser()
    args = parser.parse_args()
    input_model = onnx.load(args.input_model_path)
    onnx.checker.check_model(input_model)
    new_onnx_model = decompose_lstm(input_model)
    onnx.save(new_onnx_model, args.output_model_path)


if __name__ == "__main__":
    main()
