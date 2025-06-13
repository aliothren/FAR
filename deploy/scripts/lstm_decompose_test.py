import itertools
import time
import unittest

import numpy as np
import onnx
import onnx.reference
import onnxruntime as ort

from lstm_decompose import (
    lstm_forward_model,
    lstm_bidirectional_model
)

SEQ_LENGTH_TESTS = [1, 197, 16, 32, 64, 128]
BATCH_SIZE_TESTS = [1]
# Smaller input size could increase error
INPUT_SIZE_TESTS = [64]
HIDDEN_SIZE_TESTS = [64, 128]


class ConstructLSTMTest(unittest.TestCase):
    """Test cases on constructing a decomposed LSTM"""
    def test_check_lstm_model(self):
        """Test LSTM constructed with random weight, recurrence and bias tensors"""
        num_directions = 2
        np.random.seed(0)
        W = np.random.randn(num_directions, 4 * HIDDEN_SIZE_TESTS[0], 16).astype(np.float32)
        R = np.random.randn(num_directions, 4 * HIDDEN_SIZE_TESTS[0], HIDDEN_SIZE_TESTS[0]).astype(
            np.float32
        )
        B = np.random.randn(num_directions, 8 * HIDDEN_SIZE_TESTS[0]).astype(np.float32)
        constructed_lstm = lstm_bidirectional_model(W, R, B).to_model_proto()
        onnx.checker.check_model(constructed_lstm)
        onnx.save(constructed_lstm, "decomposed_lstm.onnx")

    def test_lstm_input_output_shapes(self):
        """Test LSTM input and output shapes match expected input and output shapes"""
        np.random.seed(0)
        test_cases = list(
            itertools.product(
                SEQ_LENGTH_TESTS, BATCH_SIZE_TESTS, INPUT_SIZE_TESTS, HIDDEN_SIZE_TESTS
            )
        )
        num_directions = 2
        for i, case in enumerate(test_cases):
            seq_length, batch_size, input_size, hidden_size = case
            W = np.random.randn(num_directions, 4 * hidden_size, input_size).astype(
                np.float32
            )
            R = np.random.randn(num_directions, 4 * hidden_size, hidden_size).astype(
                np.float32
            )
            B = np.random.randn(num_directions, 8 * hidden_size).astype(np.float32)
            constructed_lstm = lstm_bidirectional_model(W, R, B).to_model_proto()
            constructed_lstm_bytes = constructed_lstm.SerializeToString()
            constructed_lstm_session = ort.InferenceSession(
                constructed_lstm_bytes,
                sess_options=ort.SessionOptions()
                )
            print(
                "Test Case {}: seq_length={}, input_size={}, hidden_size={}".format(
                    i + 1, seq_length, input_size, hidden_size
                )
            )
            X = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
            initial_h = np.random.randn(num_directions, batch_size, hidden_size).astype(
                np.float32
            )
            initial_c = np.random.randn(num_directions, batch_size, hidden_size).astype(
                np.float32
            )
            inputs = {"X": X, "initial_h": initial_h, "initial_c": initial_c}
            Y, Y_h, Y_c = constructed_lstm_session.run(None, inputs)
            self.assertSequenceEqual(
                Y.shape,
                [
                    seq_length,
                    num_directions,
                    batch_size,
                    hidden_size,
                ],
            )
            self.assertSequenceEqual(
                Y_h.shape,
                [
                    num_directions,
                    batch_size,
                    hidden_size,
                ],
            )
            self.assertSequenceEqual(
                Y_c.shape,
                [
                    num_directions,
                    batch_size,
                    hidden_size,
                ],
            )
        print("Testing finished.")

    def test_lstm_inference(self):
        """Test LSTM inference, ensuring output tensors within 1e-04 absolute and
        1e-04 relative tolerance compared to ONNX LSTM
        Expected tolerance between 1e-04 and 1e-05
        Small inputs/weights could also increase error"""
        np.random.seed(27)
        test_cases = list(
            itertools.product(
                SEQ_LENGTH_TESTS, BATCH_SIZE_TESTS, INPUT_SIZE_TESTS, HIDDEN_SIZE_TESTS
            )
        )
        num_directions = 2
        for i, case in enumerate(test_cases):
            seq_length, batch_size, input_size, hidden_size = case
            print(
                "Test Case {}: seq_length={}, input_size={}, hidden_size={}".format(
                    i + 1, seq_length, input_size, hidden_size
                )
            )
            # Define input and output tensors
            X = onnx.helper.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, [seq_length, batch_size, input_size]
            )
            initial_h = onnx.helper.make_tensor_value_info(
                "initial_h",
                onnx.TensorProto.FLOAT,
                [num_directions, batch_size, hidden_size],
            )
            initial_c = onnx.helper.make_tensor_value_info(
                "initial_c",
                onnx.TensorProto.FLOAT,
                [num_directions, batch_size, hidden_size],
            )
            Y = onnx.helper.make_tensor_value_info(
                "Y",
                onnx.TensorProto.FLOAT,
                [seq_length, num_directions, batch_size, hidden_size],
            )
            Y_h = onnx.helper.make_tensor_value_info(
                "Y_h", onnx.TensorProto.FLOAT, [num_directions, batch_size, hidden_size]
            )
            Y_c = onnx.helper.make_tensor_value_info(
                "Y_c", onnx.TensorProto.FLOAT, [num_directions, batch_size, hidden_size]
            )
            # Define the weight tensors (W, R, B) as constants
            W = np.random.normal(
                0, 5, [num_directions, 4 * hidden_size, input_size]
            ).astype(np.float32)
            R = np.random.normal(
                0, 5, [num_directions, 4 * hidden_size, hidden_size]
            ).astype(np.float32)
            B = np.random.normal(0, 5, [num_directions, 8 * hidden_size]).astype(
                np.float32
            )
            # Create ONNX constant nodes for W, R, and B
            W_const = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["W"],
                value=onnx.numpy_helper.from_array(W, name="W"),
            )

            R_const = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["R"],
                value=onnx.numpy_helper.from_array(R, name="R"),
            )

            B_const = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["B"],
                value=onnx.numpy_helper.from_array(B, name="B"),
            )

            # Create LSTM node
            lstm_node = onnx.helper.make_node(
                "LSTM",
                inputs=["X", "W", "R", "B", "", "initial_h", "initial_c"],
                outputs=["Y", "Y_h", "Y_c"],
                hidden_size=hidden_size,
                direction="bidirectional",
            )

            # Create graph
            graph = onnx.helper.make_graph(
                nodes=[W_const, R_const, B_const, lstm_node],
                name="LSTM_Model",
                inputs=[X, initial_h, initial_c],
                outputs=[Y, Y_h, Y_c],
            )
            # Create model
            onnx_lstm = onnx.helper.make_model(
                graph,
                producer_name="onnx-LSTM",
                opset_imports=[onnx.helper.make_opsetid("", 14)],
                ir_version=6, 
            )
            onnx_lstm_bytes = onnx_lstm.SerializeToString()
            onnx_lstm_session = ort.InferenceSession(
                onnx_lstm_bytes,
                sess_options=ort.SessionOptions(),
                providers=["CPUExecutionProvider"],
                )

            constructed_lstm = lstm_bidirectional_model(W, R, B).to_model_proto()
            constructed_lstm_bytes = constructed_lstm.SerializeToString()
            constructed_lstm_session = ort.InferenceSession(
                constructed_lstm_bytes,
                sess_options=ort.SessionOptions()
                )

            X = np.random.normal(0, 5, [seq_length, batch_size, input_size]).astype(
                np.float32
            )
            initial_h = np.random.normal(
                0, 5, [num_directions, batch_size, hidden_size]
            ).astype(np.float32)
            initial_c = np.random.normal(
                0, 5, [num_directions, batch_size, hidden_size]
            ).astype(np.float32)
            inputs = {"X": X, "initial_h": initial_h, "initial_c": initial_c}
            start_time = time.time()
            Y_original, Y_h_original, Y_c_original = onnx_lstm_session.run(None, inputs)
            print(f"Original model inference time: {time.time() - start_time} seconds")
            start_time = time.time()
            Y_constructed, Y_h_constructed, Y_c_constructed = (
                constructed_lstm_session.run(None, inputs)
            )
            print(
                f"Constructed model inference time: {time.time() - start_time} seconds"
            )
            np.testing.assert_allclose(
                Y_constructed,
                Y_original,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Y mismatch in test case {i + 1}",
            )
            np.testing.assert_allclose(
                Y_h_constructed,
                Y_h_original,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Y_h mismatch in test case {i + 1}",
            )
            np.testing.assert_allclose(
                Y_c_constructed,
                Y_c_original,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Y_h mismatch in test case {i + 1}",
            )
            print(f"Test Case {i + 1} passed")
        print("Testing finished.")
