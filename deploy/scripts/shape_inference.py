"""ONNX shape inference that supports custom operators.

This is all kind of a workaround, since we define custom ops in a way
that doesnt support attaching shape inference for ONNX or ONNXRuntime to
use. So this code wraps the base ONNX shape inference in a way we can
extend to our `tetramem_experimental` ops.
"""

import typing

import onnx
import onnx.shape_inference

import model as model_lib


class ShapeInference:
    """Context for the custom shape inference."""

    def __init__(self, model: onnx.ModelProto, strict_mode: bool = False):
        self.model = model
        self.initializers = model_lib.get_initializer_map(model)
        self.functions = {
            (func.domain, func.name): func for func in self.model.functions
        }
        self.used_dim_names = self._unknown_dims_in_values(model.graph.input)
        self.unknown_counter = 0
        self.strict_mode = strict_mode

    def input_value_infos(self) -> typing.Dict[str, onnx.ValueInfoProto]:
        value_dict = {}
        for input_value in self.model.graph.input:
            value_dict[input_value.name] = input_value
        return value_dict

    @staticmethod
    def _unknown_dims_in_value(value: onnx.ValueInfoProto) -> typing.Set[str]:
        """Get all the unknown/named dimensions in a value's shape."""
        unknowns = set()
        for dim in model_lib.value_shape(value):
            if isinstance(dim, str):
                unknowns.add(dim)
        return unknowns

    @staticmethod
    def _unknown_dims_in_values(
        values: typing.Iterable[onnx.ValueInfoProto],
    ) -> typing.Set[str]:
        """Get all the unknown/named dimensions in multiple values."""
        unknowns = set()
        for value in values:
            unknowns.update(ShapeInference._unknown_dims_in_value(value))
        return unknowns

    def _rename_unknown(
        self, old_name: str, in_value_infos: typing.Iterable[onnx.ValueInfoProto]
    ):
        """Replaces instances of old_name as a dim with a unique new name."""
        new_name = f"unknown_dim_{self.unknown_counter}"
        assert new_name not in self.used_dim_names
        self.used_dim_names.add(new_name)
        self.unknown_counter += 1

        for value_info in in_value_infos:
            if not value_info.type.HasField("tensor_type"):
                continue
            shape = value_info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField("dim_param") and dim.dim_param == old_name:
                    dim.dim_param = new_name

    def _get_value_or_initializer_elem_type(
        self, value_name: str, prev_values: typing.Mapping[str, onnx.ValueInfoProto]
    ) -> int:
        if value_name in prev_values:
            value_info = prev_values[value_name]
            if not value_info.type.HasField("tensor_type"):
                raise NotImplementedError(
                    f"Only tensor types are supported, for value {value_name}"
                )
            return value_info.type.tensor_type.elem_type
        elif value_name in self.initializers:
            return self.initializers[value_name].data_type
        else:
            raise KeyError(f"No value found for {value_name}")

    def _standard_op_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        # In order to run the stock shape inference on a single node, this function
        # wraps it in a model consisting of only that node.
        node_input_values = []
        node_initializers = []
        for input_name in node_proto.input:
            if input_name == "":
                continue  # Skip empty optional inputs
            elif input_name in prev_values:
                node_input_values.append(prev_values[input_name])
            elif input_name in self.initializers:
                node_initializers.append(self.initializers[input_name])
            else:
                raise ValueError("Input to node not found in previous values.")

        # The output values are set to have unknown shape.
        node_output_values = []
        for output_name in node_proto.output:
            node_output_values.append(
                onnx.helper.make_tensor_value_info(
                    output_name, onnx.TensorProto.UNDEFINED, None
                )
            )

        graph = onnx.helper.make_graph(
            nodes=[node_proto],
            name="shape-inference-node-model",
            inputs=node_input_values,
            outputs=node_output_values,
            initializer=node_initializers,
        )
        node_model = onnx.helper.make_model(
            graph,
            opset_imports=self.model.opset_import,
            functions=self.model.functions,
        )

        # Run the base ONNX shape inference.
        inferred_model = onnx.shape_inference.infer_shapes(
            node_model,
            strict_mode=self.strict_mode,
        )

        output_values = {}
        for output in inferred_model.graph.output:
            output_values[output.name] = output

        # If the node introduces any additional names for unknown values, make
        # sure this name is unique across the rest of the model.
        input_unknowns = self._unknown_dims_in_values(node_input_values)
        output_unknowns = self._unknown_dims_in_values(output_values.values())
        new_unknowns = output_unknowns.difference(input_unknowns)
        for new_unknown in new_unknowns:
            if new_unknown in self.used_dim_names:
                self._rename_unknown(new_unknown, output_values.values())
            else:
                self.used_dim_names.add(new_unknown)

        return output_values

    def _signed_shift_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        # We assume the same shape propagates from the first input
        lhs_name = node_proto.input[0]
        lhs_value = prev_values[lhs_name]
        output_name = node_proto.output[0]
        output_value = onnx.ValueInfoProto()
        output_value.CopyFrom(lhs_value)
        output_value.name = output_name
        return {output_name: output_value}

    def _qlinear_global_average_pool_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        X_name = node_proto.input[0]
        X_value = prev_values[X_name]

        attributes = model_lib.get_attributes_map(node_proto)
        if "channels_last" not in attributes:
            raise ValueError("channels_last attribute not found")
        channels_last = attributes["channels_last"]
        input_dims = model_lib.value_shape(X_value)
        channels_index = len(input_dims) - 1 if channels_last else 1
        output_dims = [
            dim if index in [0, channels_index] else 1
            for index, dim in enumerate(input_dims)
        ]

        output_name = node_proto.output[0]
        output_value = onnx.helper.make_tensor_value_info(
            output_name, X_value.type.tensor_type.elem_type, output_dims
        )
        return {output_name: output_value}

    def _qlinear_average_pool_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Shape inference for QLinearAveragePool ONNXRuntime contrib operator.

        The output shape is computed using the standard pooling formula:
        Output_size = (Input_size - Kernel_size + two_side_Padding) / Stride + 1

        https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAveragePool
        """

        X_name = node_proto.input[0]
        X_value = prev_values[X_name]

        attributes = model_lib.get_attributes_map(node_proto)
        if "kernel_shape" not in attributes:
            raise ValueError("Missing required pooling attributes: kernel_shape")

        kernel_shape = attributes["kernel_shape"]
        strides = attributes["strides"] if "strides" in attributes else [1, 1]
        # [pad_h_before, pad_w_before, pad_h_after, pad_w_after]
        pads = attributes["pads"] if "pads" in attributes else [0, 0, 0, 0]

        if len(pads) != 4:
            raise ValueError(f"Expected 4 padding values, but got {len(pads)}")

        pad_h_before, pad_w_before, pad_h_after, pad_w_after = pads

        input_dims = model_lib.value_shape(X_value)
        batch_size, channels = input_dims[0], input_dims[1]
        input_height, input_width = input_dims[2], input_dims[3]

        output_height = (
            input_height - kernel_shape[0] + pad_h_before + pad_h_after
        ) // strides[0] + 1
        output_width = (
            input_width - kernel_shape[1] + pad_w_before + pad_w_after
        ) // strides[1] + 1

        output_dims = [batch_size, channels, output_height, output_width]
        output_name = node_proto.output[0]
        output_type = X_value.type.tensor_type.elem_type

        output_value = onnx.helper.make_tensor_value_info(
            output_name, output_type, output_dims
        )
        return {output_name: output_value}

    def _qgemm_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Shape inference for the QGemm ONNXRuntime contrib operator.

        https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QGemm
        """
        # TODO: Support A and B being either inputs or initializers.
        A_name = node_proto.input[0]
        A_value = prev_values[A_name]
        B_name = node_proto.input[3]
        B_value = self.initializers[B_name]

        attributes = model_lib.get_attributes_map(node_proto)
        transA = attributes["transA"] if "transA" in attributes else 0
        transB = attributes["transB"] if "transB" in attributes else 0
        # TODO: Support multiplication of values with ranks other than 2.
        dim0 = model_lib.value_shape(A_value)[transA]
        dim1 = model_lib.tensor_shape(B_value)[transB ^ 1]

        if len(node_proto.input) < 9:
            # Output tyep defaults to float if there is no y_zero_point input.
            output_type = int(onnx.TensorProto.FLOAT)
        else:
            y_zero_point_name = node_proto.input[8]
            output_type = self._get_value_or_initializer_elem_type(
                y_zero_point_name, prev_values
            )
        output_name = node_proto.output[0]

        output_value = onnx.helper.make_tensor_value_info(
            output_name, output_type, [dim0, dim1]
        )
        return {output_name: output_value}

    def _qlinear_binary_elementwise_inference(
        self,
        input_names: typing.Sequence[str],
        output_name: str,
        base_op_type: str,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        base_op = onnx.helper.make_node(
            op_type=base_op_type,
            inputs=input_names,
            outputs=[output_name],
        )
        return self._standard_op_shape_inference(base_op, prev_values)

    def _qlinearadd_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Shape inference for QLinearAdd ONNXRuntime contrib operator.

        Uses the same shape inference math as the basic "Add" op.

        https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAdd
        """
        if node_proto.domain != "com.microsoft" or node_proto.op_type != "QLinearAdd":
            raise ValueError("Expected a QLinearAdd op")
        if len(node_proto.input) != 8:
            raise ValueError(
                f"Expected exactly 8 inputs, got {len(node_proto.input)}."
                " Eliding optional arguments not yet supported"
            )
        if len(node_proto.output) != 1:
            raise ValueError(
                f"Expected exactly one output, got {len(node_proto.output)}"
            )
        mul_inputs = [node_proto.input[0], node_proto.input[3]]
        return self._qlinear_binary_elementwise_inference(
            input_names=mul_inputs,
            output_name=node_proto.output[0],
            base_op_type="Add",
            prev_values=prev_values,
        )

    def _qlinearconcat_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ):
        """Shape inference for QLinearConcat ONNXRuntime contrib operator.

        Uses the same shape inference math as the basic "Concat" op.

        https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearConcat
        """
        if (
            node_proto.domain != "com.microsoft"
            or node_proto.op_type != "QLinearConcat"
        ):
            raise ValueError("Expected a QLinearConcat op")

        # https://github.com/microsoft/onnxruntime/blob/3e4c5e64877c6d9814e4ebce5dcbb1fe71588ec5/onnxruntime/contrib_ops/cpu/quantization/qlinear_concat.cc#L18
        if len(node_proto.input) < 5:
            raise ValueError(
                "Expected at least 5 inputs from QLinearConcat, got "
                f"{len(node_proto.input)}"
            )
        if (len(node_proto.input) - 2) % 3 != 0:
            raise ValueError(
                "Expected each QLinearConcat input to be (input, scale, zero "
                f"point), triplet. Got {len(node_proto.input)} inputs"
            )
        if len(node_proto.output) != 1:
            raise ValueError(
                "Expected exactly 1 output from QLinearConcat, got "
                f"{len(node_proto.output)}"
            )

        input_count = (len(node_proto.input) - 2) // 3
        input_names = [
            node_proto.input[2 + 3 * input_index] for input_index in range(input_count)
        ]
        output_name = node_proto.output[0]

        attributes = model_lib.get_attributes_map(node_proto)
        if "axis" not in attributes:
            raise ValueError('Required attribute "axis" missing from QLinearConcat')

        base_op = onnx.helper.make_node(
            op_type="Concat",
            inputs=input_names,
            outputs=[output_name],
            axis=attributes["axis"],
        )
        return self._standard_op_shape_inference(base_op, prev_values)

    def _qlinearmul_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Shape inference for QLinearMul ONNXRuntime contrib operator.

        Uses the same shape inference math as the basic "Mul" op.

        https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearMul
        """
        if node_proto.domain != "com.microsoft" or node_proto.op_type != "QLinearMul":
            raise ValueError("Expected a QLinearMul op")
        if len(node_proto.input) != 8:
            raise ValueError(
                f"Expected exactly 8 inputs, got {len(node_proto.input)}."
                " Eliding optional arguments not yet supported"
            )
        if len(node_proto.output) != 1:
            raise ValueError(
                f"Expected exactly one output, got {len(node_proto.output)}"
            )
        mul_inputs = [node_proto.input[0], node_proto.input[3]]
        return self._qlinear_binary_elementwise_inference(
            input_names=mul_inputs,
            output_name=node_proto.output[0],
            base_op_type="Mul",
            prev_values=prev_values,
        )

    def _qlinearsigmoid_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Shape inference for QLinearSigmoid ONNXRuntime contrib operator.

        Output is expected to have the same shape and type as the input.

        https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSigmoid
        """
        input_name = node_proto.input[0]
        output_name = node_proto.output[0]
        output_value_info = onnx.ValueInfoProto()
        output_value_info.CopyFrom(prev_values[input_name])
        output_value_info.name = output_name
        return {output_name: output_value_info}

    def _qlinearleakyrelu_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Shape inference for QLinearLeakyRelu ONNXRuntime contrib operator.

        Output is expected to have the same shape and type as the input.

        https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftqlinearleakyrelu
        """
        input_name = node_proto.input[0]
        output_name = node_proto.output[0]
        output_value_info = onnx.ValueInfoProto()
        output_value_info.CopyFrom(prev_values[input_name])
        output_value_info.name = output_name
        return {output_name: output_value_info}

    def _piecewise_linear_integer_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        if len(node_proto.input) != 3:
            raise ValueError("Expected exactly 3 inputs to PiecewiseLinearInteger node")
        if len(node_proto.output) != 1:
            raise ValueError(
                "Expected exactly 1 outputs from PiecewiseLinearInteger node"
            )

        # We assume the same shape propagates from the first input
        lhs_name = node_proto.input[0]
        lhs_value = prev_values[lhs_name]
        output_name = node_proto.output[0]
        output_value = onnx.ValueInfoProto()
        output_value.CopyFrom(lhs_value)
        output_value.name = output_name

        # But the element type will come from the breakpoints_y input
        breakpoints_y_name = node_proto.input[2]
        if breakpoints_y_name in self.initializers:
            breakpoints_y_tensor = self.initializers[breakpoints_y_name]
            output_value.type.tensor_type.elem_type = breakpoints_y_tensor.data_type
        else:
            breakpoints_y_value = prev_values[breakpoints_y_name]
            output_value.type.tensor_type.elem_type = (
                breakpoints_y_value.type.tensor_type.elem_type
            )

        return {output_name: output_value}

    def _shale_conv_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Computes shape inference on the Tetramem custom ShaleConv op.

        This op is meant to represent integer convolution, with individual VMMs
        computed in a simulation of the Shale NPU. The shape inference is the
        same as ConvInteger, and ShaleConv has the same attributes for padding,
        strides, etc. (Though the handling of optional attributes is different.)
        """
        if node_proto.op_type != "ShaleConv":
            raise ValueError("Expected a ShaleConv node.")

        attrs = model_lib.get_attributes_map(node_proto)
        if "auto_pad" not in attrs or attrs["auto_pad"] == b"NOTSET":
            pads = attrs["pads"]
        else:
            pads = None
        attrs.pop("pads")

        # Do inference on an equivalent ConvInteger.
        standard_node = onnx.helper.make_node(
            op_type="ConvInteger",
            inputs=node_proto.input,
            outputs=node_proto.output,
            pads=pads,
            **attrs,
        )
        return self._standard_op_shape_inference(standard_node, prev_values)

    def _shale_mat_mul_shape_inference(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        """Computes shape inference on the Tetramem custom ShaleMatMul op.

        This op is meant to represent integer matrix multiplication, where the
        computation is done in a simulation of the ShaleNPU. The shape inference
        is the same as MatMulInteger, and so this is re-used from the base ONNX
        library.
        """
        if node_proto.op_type != "ShaleMatMul":
            raise ValueError("Expected a ShaleMatMul node.")

        # Do inference on an equivalent MatMulInteger.
        standard_node = onnx.helper.make_node(
            op_type="MatMulInteger",
            inputs=node_proto.input,
            outputs=node_proto.output,
        )
        return self._standard_op_shape_inference(standard_node, prev_values)

    def _get_copy_input_shape_handler(self, output_type: typing.Optional[int] = None):
        """Create handler assuming the output shape matches the first input.

        The output type is dynamic unless specified.
        """

        def shape_inference_func(
            node_proto: onnx.NodeProto,
            prev_values: typing.Mapping[str, onnx.ValueInfoProto],
        ) -> typing.Dict[str, onnx.ValueInfoProto]:
            """Infers the output shape to match the input."""

            lhs_name = node_proto.input[0]
            lhs_value = prev_values[lhs_name]
            output_name = node_proto.output[0]
            output_value = onnx.ValueInfoProto()
            output_value.CopyFrom(lhs_value)
            output_value.name = output_name
            output_value.type.tensor_type.elem_type = (
                output_type or lhs_value.type.tensor_type.elem_type
            )
            return {output_name: output_value}

        return shape_inference_func

    def _get_equivalent_inputs_handler(
        self,
        custom_op_type: str,
        standard_op_type: str,
        explicit_output_type: typing.Optional[int] = None,
    ):
        """Create handler assuming that there is a standard op that
        has the same inputs.

        Operations are expected to have a single output. Its type will
        match the first input of the node.
        """

        def shape_inference_func(
            node_proto: onnx.NodeProto,
            prev_values: typing.Mapping[str, onnx.ValueInfoProto],
        ) -> typing.Dict[str, onnx.ValueInfoProto]:
            if node_proto.op_type != custom_op_type:
                raise ValueError(f"Expected a {custom_op_type} node.")
            # First do inference on an equivalent standard op.
            # The result should be the same, except for the output type.
            standard_node = onnx.NodeProto()
            standard_node.CopyFrom(node_proto)
            standard_node.op_type = standard_op_type
            # Use default domain (that includes standard op)
            standard_node.domain = ""
            standard_output_values = self._standard_op_shape_inference(
                standard_node, prev_values
            )

            if len(standard_output_values) != 1:
                raise ValueError(
                    f"""Expected exactly one output from {standard_op_type},
                    {len(standard_output_values)} outputs found"""
                )
            if len(node_proto.output) != 1:
                raise ValueError(
                    f"""Expected exactly one output from {custom_op_type},
                    {len(node_proto.output)} outputs found"""
                )

            output_name = node_proto.output[0]
            output_value = standard_output_values[output_name]
            if not explicit_output_type:
                X_name = node_proto.input[0]
                X_value = prev_values[X_name]
                output_type = X_value.type.tensor_type.elem_type
            else:
                output_type = explicit_output_type
            output_value.type.tensor_type.elem_type = output_type

            return {output_name: output_value}

        return shape_inference_func

    @property
    def inference_methods(self):
        return {
            "NPUConv": self._get_equivalent_inputs_handler("NPUConv", "ConvInteger"),
            "NPUMatMul": self._get_equivalent_inputs_handler(
                "NPUMatMul", "MatMulInteger"
            ),
            "SignedBitShift": self._signed_shift_shape_inference,
            "ShaleConv": self._shale_conv_shape_inference,
            "ShaleMatMul": self._shale_mat_mul_shape_inference,
            "NormBlock": self._get_copy_input_shape_handler(),
            "PiecewiseLinear": self._get_copy_input_shape_handler(),
            "PiecewiseLinearInteger": self._piecewise_linear_integer_shape_inference,
            "QLinearGlobalAveragePool": (
                self._qlinear_global_average_pool_shape_inference
            ),
            "QLinearAdd": self._qlinearadd_shape_inference,
            "QLinearConcat": self._qlinearconcat_shape_inference,
            "QLinearMul": self._qlinearmul_shape_inference,
            "QLinearSigmoid": self._qlinearsigmoid_shape_inference,
            "QLinearSoftmax": self._get_copy_input_shape_handler(),
            "QGemm": self._qgemm_shape_inference,
            "QLinearLeakyRelu": self._qlinearleakyrelu_shape_inference,
            "QLinearAveragePool": self._qlinear_average_pool_shape_inference,
        }

    def node_output_shapes(
        self,
        node_proto: onnx.NodeProto,
        prev_values: typing.Mapping[str, onnx.ValueInfoProto],
    ) -> typing.Dict[str, onnx.ValueInfoProto]:
        if not node_proto.domain:
            return self._standard_op_shape_inference(node_proto, prev_values)
        # Also hands off functions to the standard shape inference. This won't
        # yet support functions that include custom ops inside. (But will
        # support custom-defined functions.)
        if (node_proto.domain, node_proto.op_type) in self.functions:
            return self._standard_op_shape_inference(node_proto, prev_values)
        if (node_proto.domain in ["tetramem_experimental", "com.microsoft"]) and (
            node_proto.op_type in self.inference_methods
        ):
            try:
                return self.inference_methods[node_proto.op_type](
                    node_proto, prev_values
                )
            except ValueError as e:
                if self.strict_mode:
                    raise e
                else:
                    return {}

        if self.strict_mode:
            raise ValueError(
                f"Unrecognized op {node_proto.domain}::{node_proto.op_type}"
            )
        else:
            return {}

    def infer_shapes(self) -> onnx.ModelProto:
        # Do shape inference node-by-node to get the shapes of all values.
        value_infos = self.input_value_infos()
        computed_value_names = []
        for node in self.model.graph.node:
            value_infos.update(self.node_output_shapes(node, value_infos))
            computed_value_names.extend(node.output)

        # Save the inferred shapes in an output ModelProto.
        inferred_model = onnx.ModelProto()
        inferred_model.CopyFrom(self.model)

        for output_value in inferred_model.graph.output:
            if output_value.name not in value_infos:
                if self.strict_mode:
                    raise ValueError(f"Did not infer a shape for {output_value.name}")
                else:
                    continue
            output_value.CopyFrom(value_infos[output_value.name])

        del inferred_model.graph.value_info[:]
        model_output_names = {
            output_value.name for output_value in self.model.graph.output
        }
        for computed_value_name in computed_value_names:
            if computed_value_name in model_output_names:
                continue
            inferred_model.graph.value_info.append(value_infos[computed_value_name])

        return inferred_model


def infer_shapes(model: onnx.ModelProto, strict_mode: bool = False) -> onnx.ModelProto:
    """ONNX shape inference that supports some custom operators.

    If strict_mode is True, will throw an error if a malformed graph prevents
    shape inference.
    """
    return ShapeInference(model, strict_mode=strict_mode).infer_shapes()
