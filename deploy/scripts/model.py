""""Code for working with ONNX models."""

import typing
from dataclasses import dataclass

import numpy as np
import onnx
import onnx.numpy_helper


def get_attributes_map(node: onnx.NodeProto) -> typing.Dict[str, typing.Any]:
    """Gets a dict mapping attribute names to a python value.

    See
    https://github.com/onnx/onnx/blob/main/docs/Operators.md
    for the expected attributes for different operators.
    """
    attr_map = {}
    for attribute in node.attribute:
        if not attribute.name:
            raise ValueError("Unnamed node attribute.")
        if attribute.name in attr_map:
            raise ValueError(f'Repeated attribute name "{attribute.name}"')
        attr_map[attribute.name] = onnx.helper.get_attribute_value(attribute)
    return attr_map


_CONSTANT_ATTRS = [
    "value",
    "value_float",
    "value_floats",
    "value_int",
    "value_ints",
    "value_string",
    "value_strings",
]


_CONSTANT_ATTR_TO_DTYPE = {
    "value_float": onnx.TensorProto.FLOAT,
    "value_floats": onnx.TensorProto.FLOAT,
    "value_int": onnx.TensorProto.INT64,
    "value_ints": onnx.TensorProto.INT64,
    "value_string": onnx.TensorProto.STRING,
    "value_strings": onnx.TensorProto.STRING,
}


def get_constant_map(model: onnx.ModelProto) -> typing.Dict[str, onnx.TensorProto]:
    """Gets a dict mapping constant tensor names to their `TensorProto`.

    ONNX may define nodes with op_type "Constant" that will contain a
    value. This helper function retrieves those values.
    """
    graph = model.graph
    constant_map = {}
    for node in graph.node:
        if node.op_type == "Constant":
            constant_name = node.output[0]
            attrs = get_attributes_map(node)
            for attr_name in _CONSTANT_ATTRS:
                if attr_name in attrs:
                    constant_tensor = attrs[attr_name]
                    break
            else:
                raise ValueError("Unhandled Constant attribute.")
            if attr_name != "value":
                tensor_dtype = _CONSTANT_ATTR_TO_DTYPE[attr_name]
                if attr_name.endswith("s"):
                    constant_tensor = onnx.helper.make_tensor(
                        constant_name,
                        tensor_dtype,
                        dims=[len(constant_tensor)],
                        vals=constant_tensor,
                    )
                else:
                    constant_tensor = onnx.helper.make_tensor(
                        constant_name, tensor_dtype, dims=[], vals=[constant_tensor]
                    )
            constant_tensor.name = constant_name
            constant_map[constant_name] = constant_tensor
    return constant_map


def get_initializer_map(model: onnx.ModelProto) -> typing.Dict[str, onnx.TensorProto]:
    """Gets a dict mapping initializer tensor names to their `TensorProto`.

    Initializers are those tensors, such as weights, that are not
    expected to vary with each run of the model. This assumes that each
    initializer has a name set, and provides a way to look them up by
    name.
    """
    graph = model.graph
    if len(graph.sparse_initializer):
        raise NotImplementedError("Sparse initializers not supported.")
    init_map = {}
    for tensor in graph.initializer:
        if not tensor.name:
            raise ValueError("Initializer without name.")
        if tensor.name in init_map:
            raise ValueError(f'Repeated initializer name "{tensor.name}"')
        init_map[tensor.name] = tensor
    return init_map


def get_value_info_map(model: onnx.ModelProto) -> typing.Dict[str, onnx.ValueInfoProto]:
    """Maps names to `ValuesInfoProto`s for all tensors in the model.

    These `ValueInfoProto`s describe the superset containing all of:
    1. Model inputs
    2. Model outputs
    3. Intermediate tensors computed by the model.
    """
    graph = model.graph
    model_input_infos = {input_info.name: input_info for input_info in graph.input}
    model_output_infos = {output_info.name: output_info for output_info in graph.output}
    # Running shape inference ensures that the `ValueInfo` for intermediate
    # tensors is populated.
    inferred_model = onnx.shape_inference.infer_shapes(model)
    inferred_graph = inferred_model.graph
    model_intermediate_infos = {
        value_info.name: value_info for value_info in inferred_graph.value_info
    }

    all_value_infos = {}
    all_value_infos.update(model_input_infos)
    all_value_infos.update(model_output_infos)
    all_value_infos.update(model_intermediate_infos)
    return all_value_infos


def tensor_shape(tensor_proto: onnx.TensorProto) -> typing.List[int]:
    """Gets the shape from an ONNX `TensorProto` as a simple list."""
    return list(tensor_proto.dims)


def value_shape(
    value_proto: onnx.ValueInfoProto,
) -> typing.List[typing.Union[int, str]]:
    """Gets the shape from an ONNX `ValueInfo` as a simple list."""
    if value_proto.type.HasField("tensor_type"):
        return [
            dim.dim_value or dim.dim_param
            for dim in value_proto.type.tensor_type.shape.dim
        ]
    raise ValueError("Unsupported value type.")


@dataclass
class NodeWithWeights:
    """An ONNX op node and its input weights/params."""

    node_proto: onnx.NodeProto
    # `weights` maps from tensor names to their protos.
    weights: typing.Mapping[str, onnx.TensorProto]

    # Info on inputs and outputs.
    inputs: typing.Mapping[str, onnx.ValueInfoProto]
    outputs: typing.Mapping[str, onnx.ValueInfoProto]

    @property
    def name(self) -> str:
        """The name of this node, from the NodeProto field."""
        return self.node_proto.name

    def weight_as_array(self, tensor_name: str) -> np.ndarray:
        """Gets a weight tensor as a numpy array."""
        return onnx.numpy_helper.to_array(self.weights[tensor_name])

    @property
    def op_type(self) -> str:
        """Get the type of operation done by the node (e.g. Mul, Conv, etc)."""
        return self.node_proto.op_type

    def short_description(self) -> str:
        """Creates a string with a short description of the node.

        We can use the "name" field to identify the node, but this field
        is optional. So if it's not present, this uses the op_type to at
        least give some hints in user-visible messages of what node is
        being referred to.
        """
        if not self.op_type:
            return "untyped node"  # This is probaby an error
        if self.name:
            return f'{self.op_type} node "{self.name}"'
        return f"{self.op_type} node"

    def weight_shape(self, weight_name: str) -> typing.List[int]:
        """Gets the shape of a weight tensor for this node."""
        return tensor_shape(self.weights[weight_name])

    def input_shape(self, value_name: str) -> typing.List[typing.Union[int, str]]:
        """Gets the shape of an input value for this node."""
        return value_shape(self.inputs[value_name])

    def output_shape(self, value_name: str) -> typing.List[typing.Union[int, str]]:
        """Gets the shape of an output value for this node."""
        return value_shape(self.outputs[value_name])

    def node_attributes(self) -> typing.Dict[str, typing.Any]:
        return get_attributes_map(self.node_proto)


_NODE_WITH_WEIGHTS_SKIP_OPS = {"Constant"}


def get_nodes_with_weights(model: onnx.ModelProto) -> typing.Iterable[NodeWithWeights]:
    """Gets each model node as a `NodeWithWeights`."""
    initializers = get_initializer_map(model)
    constants = get_constant_map(model)
    value_infos = get_value_info_map(model)
    for node in model.graph.node:
        if node.op_type in _NODE_WITH_WEIGHTS_SKIP_OPS:
            continue

        # Find the `TensorProto`s for this node's weights.
        node_weights = {}
        input_infos = {}
        for input_name in node.input:
            if not input_name:
                continue
            if input_name in initializers:
                node_weights[input_name] = initializers[input_name]
            elif input_name in constants:
                node_weights[input_name] = constants[input_name]
            else:
                input_infos[input_name] = value_infos[input_name]
        output_infos = {
            output_name: value_infos[output_name] for output_name in node.output
        }

        yield NodeWithWeights(
            node_proto=node,
            weights=node_weights,
            inputs=input_infos,
            outputs=output_infos,
        )


def create_single_node_graph(node: NodeWithWeights) -> onnx.GraphProto:
    """Creates an ONNX graph consisting of a single node."""
    node_name = node.node_proto.name or node.node_proto.op_type
    singleton_graph_doc_string = f'Singleton graph for "{node_name}" node'
    return onnx.helper.make_graph(
        nodes=[node.node_proto],
        name=node_name,
        inputs=list(node.inputs.values()),
        outputs=list(node.outputs.values()),
        initializer=list(node.weights.values()),
        doc_string=singleton_graph_doc_string,
    )


# TODO: This should be renamed to something other than tetramem_experimental
TETRAMEM_OPSET_NAME = "tetramem_experimental"


def tetramem_opsets() -> typing.List[onnx.OperatorSetIdProto]:
    """Gets the default opsets including TetraMem custom ops."""
    return [
        onnx.helper.make_operatorsetid("", onnx.defs.onnx_opset_version()),
        onnx.helper.make_operatorsetid(TETRAMEM_OPSET_NAME, 1),
        onnx.helper.make_operatorsetid("com.microsoft", 1),
    ]


def add_tetramem_opset(model: onnx.ModelProto):
    """Modifies the model to define the TetraMem custom op domain.

    Will be a no-op if the domain is already defined.

    """
    # Skip if this domain is already present.
    for opset_id in model.opset_import:
        if opset_id.domain == TETRAMEM_OPSET_NAME:
            return
    model.opset_import.append(onnx.helper.make_operatorsetid(TETRAMEM_OPSET_NAME, 1))
