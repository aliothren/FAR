"""Working with an ONNX model as a rustworkx PyDiGraph, and converting it.

This represents the model as a directed graph (DiGraph). It uses the rustworkx
graph library, so that we can apply the algorithms from the rustworkx library
to analyze an ONNX model.

The representation in the digraph will have a node for each operator in the
model, as well as a node for each input & output value, and each weight. The
edges of the digraph will represent the tensor values that are the input/output
of each node.
"""

import collections
import dataclasses
import enum
import operator
import os
import typing

import numpy as np
import onnx
import rustworkx as rx
import rustworkx.visualization

import shape_inference


class NodeType(enum.Enum):
    """The type of network node."""

    OPERATOR = 1
    INPUT = 2
    OUTPUT = 3
    INITIALIZER = 4


@dataclasses.dataclass
class NetworkNode:
    """Data included with"""

    type: NodeType
    name: str
    onnx_node: typing.Optional[onnx.NodeProto] = None


@dataclasses.dataclass
class NetworkEdge:
    """Data included with edges in the model network."""

    name: str

    value_info: typing.Optional[onnx.ValueInfoProto] = None

    # In the case that this value is the output of a node, this gives the index
    # into the `output` field of the NodeProto.
    producer_arg_index: typing.Optional[int] = None

    # In the case that this value is the input of a node, this gives the index
    # into the `input` field of the NodeProto.
    consumer_arg_index: typing.Optional[int] = None


class SubgraphTemplate:
    """Defines a subgraph to search for in a larger ONNX model."""

    digraph: rx.PyDiGraph

    def __init__(self):
        self.digraph = rx.PyDiGraph()

        self.input_nodes = set()
        self.output_nodes = set()

    def add_template_node(
        self,
        matcher: typing.Callable[[NetworkNode], bool] = lambda _: True,
    ) -> int:
        """Adds a network node (e.g. an operator) to be matched.

        A node will be matched only if `matcher` returns True for its
        `NetworkNode` data. Returns the index of this node in the template
        graph.
        """
        return self.digraph.add_node(matcher)

    def add_operator(
        self,
        matcher: typing.Callable[[onnx.NodeProto], bool] = lambda _: True,
    ) -> int:
        """Adds an operator node to be matched.

        This matches nodes that correspond to an ONNX NodeProto, representing
        an operator to be computed by the model. An operator will be matched
        only if `matcher` returns True for its `NodeProto` data. Returns the
        index of this node in the template graph.
        """

        def node_matcher(network_node: NetworkNode) -> bool:
            if network_node.type != NodeType.OPERATOR:
                return False
            if network_node.onnx_node is None:
                raise AssertionError("onnx_node not set for operator node")
            return matcher(network_node.onnx_node)

        return self.add_template_node(node_matcher)

    def add_operator_of_type(self, op_type: str) -> int:
        """Adds an operator to be matched iff it's the specified `op_type`."""

        def op_type_matcher(node_proto: onnx.NodeProto) -> bool:
            return node_proto.op_type == op_type

        return self.add_operator(op_type_matcher)

    def add_initializer(self) -> int:
        """Adds an initializer to be matched by the template."""

        def node_matcher(network_node: NetworkNode) -> bool:
            return network_node.type == NodeType.INITIALIZER

        return self.add_template_node(node_matcher)

    def add_template_edge(
        self,
        parent: int,
        child: int,
        matcher: typing.Callable[[NetworkEdge], bool] = lambda _: True,
    ) -> int:
        """Adds an edge between two nodes to be matched by the template.

        `parent` and `child` are indices of nodes in the template graph. A
        subgraph matched by this template must have an edge between two matched
        nodes. This edge will be matched only if `matcher` returns true for its
        `NetworkEdge` data.
        """
        return self.digraph.add_edge(parent, child, matcher)

    def add_operator_arg(
        self,
        producer: int,
        consumer: int,
        expected_input_index: int,
    ):
        """Adds an edge between a `producer` node and an operator `consumer`.

        `producer` and `consumer` are indices of nodes in the template graph.
        The edge in the model will be matched only if the value this edge
        represents is in the position of `expected_input_index` in the
        consumer's input list.
        """

        def match_input_index(edge: NetworkEdge) -> bool:
            return edge.consumer_arg_index == expected_input_index

        return self.add_template_edge(producer, consumer, match_input_index)

    def add_input_edge(
        self, child: int, expected_input_index: typing.Optional[int] = None
    ):
        """An edge from a node *not* in the template to one that is.

        Matches a edge with `parent` and `child` nodes where `parent` is not
        included in the match, but `child` is. If `expected_input_index` is
        set, also requires that the input value be at a specified position in
        `child`'s list of input values.
        """
        parent = self.add_template_node()
        self.input_nodes.add(parent)
        if expected_input_index is None:
            return self.add_template_edge(parent, child)
        else:
            return self.add_operator_arg(parent, child, expected_input_index)

    def add_output_edge(
        self, parent: int, expected_output_index: typing.Optional[int] = None
    ):
        """An edge from a node *not* in the template to one that is.

        Matches a edge with `parent` and `child` nodes where `parent` is not
        included in the match, but `child` is. If `expected_output_index` is
        set, also requires that the input value be at a specified position in
        `child`'s list of input values.
        """
        child = self.add_template_node()
        self.output_nodes.add(child)

        if expected_output_index:

            def match_edge(edge: NetworkEdge) -> bool:
                return edge.producer_arg_index == expected_output_index

            return self.add_template_edge(parent, child, match_edge)
        else:
            return self.add_template_edge(parent, child)

    def visualize_graph(self, op_types: list[str], output_path: str):
        """Visualize template graph with matcher nodes
        by using dummy nodes and graphviz.

        The op_types needs to be given to specify the operator types
        to be visualized, and output path needs to be given to
        save the graph as an image file to output path"""
        dummy_operator_nodes = []

        for op_type in op_types:
            dummy_node = onnx.helper.make_node(
                op_type,
                ["input"],
                ["output"],
            )
            dummy_operator_nodes.append(
                NetworkNode(name=op_type, type=NodeType.OPERATOR, onnx_node=dummy_node)
            )
        dummy_input_node = NetworkNode(name="input", type=NodeType.INPUT)
        dummy_output_node = NetworkNode(name="output", type=NodeType.OUTPUT)
        dummy_initializer = NetworkNode(name="initializer", type=NodeType.INITIALIZER)

        def node_attr_fn(
            node_matcher: typing.Callable[[NetworkNode], bool] = lambda _: True
        ):
            # TODO: There is a bug in matching input and output so we cannot tell
            # if it is input node or output node from node matcher,
            # currently it is fine and we will fix it in the future.
            if node_matcher(dummy_input_node) or node_matcher(dummy_output_node):
                return {
                    "shape": "ellipse",
                    "fillcolor": "coral",
                    "style": "filled",
                    "label": "S/E",
                }
            elif node_matcher(dummy_initializer):
                return {
                    "shape": "ellipse",
                    "fillcolor": "darkolivegreen1",
                    "style": "filled",
                    "label": "constant",
                }
            else:
                for dummy_operator_node in dummy_operator_nodes:
                    if (
                        node_matcher(dummy_operator_node)
                        and dummy_operator_node.onnx_node is not None
                    ):
                        return {
                            "shape": "rect",
                            "fillcolor": "cadetblue1",
                            "style": "filled",
                            "label": dummy_operator_node.onnx_node.op_type,
                        }
                return {
                    "shape": "rect",
                    "fillcolor": "azure",
                    "style": "filled",
                    "label": "NA",
                }

        rustworkx.visualization.graphviz_draw(
            self.digraph,
            node_attr_fn=node_attr_fn,
            filename=output_path,
        )


class MatchedSubgraph:
    """Defines a subgraph in a larger model matched by a `SubgraphTemplate`."""

    match_dict: typing.Dict[int, int]
    reverse_match_dict: typing.Dict[int, int]

    def __init__(
        self,
        match_dict: typing.Mapping[int, int],
        model,
        template: SubgraphTemplate,
    ):
        self.match_dict = dict(match_dict.items())
        self.reverse_match_dict = {dst: src for src, dst in match_dict.items()}
        self._model = model
        self._template = template

    def get_node_data(self, template_index: int) -> NetworkNode:
        """Gets data from a model node matched by a node in the template.

        `template_index` is the index of a node in the template graph, that has
        been matched to a node in the model graph. This will return the
        `NetworkNode` data for that node in the model.
        """
        return self._model.digraph.get_node_data(
            self.reverse_match_dict[template_index]
        )

    def get_operator_data(self, template_index: int) -> onnx.NodeProto:
        """Gets data from an operator matched by a node in the template."""
        node_data = self.get_node_data(template_index)
        if node_data.type != NodeType.OPERATOR:
            raise ValueError(f"Node matched to {template_index} was not an operator.")
        if node_data.onnx_node is None:
            raise ValueError("Missing ONNX node data.")
        return node_data.onnx_node

    def get_initializer_data(self, template_index: int) -> onnx.TensorProto:
        """Gets data from an initializer matched by a node in the template."""
        node_data = self.get_node_data(template_index)
        if node_data.type != NodeType.INITIALIZER:
            raise ValueError(
                f"Node matched to {template_index} was not an initializer."
            )
        return self._model.initializers[node_data.name]

    def get_edge_data_by_index(self, template_edge_index: int) -> NetworkEdge:
        """Gets data from a model edge matched by a node in the template.

        `template_edge_index` is the index of an edge in the template graph,
        that has been matched to an edge in the model graph. This will return the
        `NetworkEdge` data for that node in the model.
        """
        (
            template_src_index,
            template_dst_index,
        ) = self._template.digraph.get_edge_endpoints_by_index(template_edge_index)
        model_src_index = self.reverse_match_dict[template_src_index]
        model_dst_index = self.reverse_match_dict[template_dst_index]
        return self._model.digraph.get_edge_data(model_src_index, model_dst_index)

    def get_value_info(self, template_edge_index: int) -> onnx.ValueInfoProto:
        """Gets the ValueInfoProto associated with a matched edge."""
        edge_data = self.get_edge_data_by_index(template_edge_index)
        if edge_data.value_info is None:
            raise ValueError("No value info for this edge.")
        return edge_data.value_info

    def model_node_indices(self) -> typing.Generator[int, None, None]:
        """Indices of nodes in the ModelNetwork included in the match."""
        for index in self.match_dict.keys():
            # We treat the "input" and "output" nodes as not really being part
            # of the match: these are wildcard placeholder in order to match
            # the input/output *edges*.
            if self.match_dict[index] in self._template.input_nodes:
                continue
            if self.match_dict[index] in self._template.output_nodes:
                continue
            yield index


class ModelNetwork:
    """Wraps a rustworkx PyDiGraph that represents an ONNX model."""

    digraph: rx.PyDiGraph
    initializers: typing.Dict[str, onnx.TensorProto]
    value_infos: typing.Dict[str, onnx.ValueInfoProto]
    _producer_node_indices: typing.Dict[str, int]
    _producer_arg_indices: typing.Dict[str, typing.Optional[int]]

    def __init__(self, model: onnx.ModelProto):
        self.digraph = rx.PyDiGraph()
        self.initializers = {}
        self.value_infos = {}

        self.opset_import = list(model.opset_import)
        self.functions = list(model.functions)
        self.graph_name = model.graph.name

        # Maps from a value name to the index of the DiGraph node that produces
        # it.
        self._producer_node_indices = {}
        # Marks which output argument of the node produced this value.
        self._producer_arg_indices = {}

        # Want to populate the graph value_info field with all intermediate
        # values, and the shape inference does this.
        inferred_model = onnx.ModelProto()
        inferred_model.CopyFrom(model)
        del inferred_model.graph.value_info[:]
        inferred_model = shape_inference.infer_shapes(inferred_model, strict_mode=True)
        self._add_model_graph(inferred_model.graph)

    def add_value_info(self, value_info: onnx.ValueInfoProto):
        """Add a ValueInfoProto for a value in the model."""
        if not value_info.name:
            raise ValueError("Can only handled named values.")
        # TODO: if the value info is already present, should check it matches
        self.value_infos[value_info.name] = value_info

    def add_model_input(self, value_info: onnx.ValueInfoProto) -> int:
        """Adds an input value of an ONNX model to the DiGraph."""
        if not value_info.name:
            raise ValueError("Inputs must be named.")
        name = value_info.name
        self.add_value_info(value_info)

        payload = NetworkNode(type=NodeType.INPUT, name=name)
        node_index = self.digraph.add_node(payload)
        self._producer_node_indices[name] = node_index
        self._producer_arg_indices[name] = None

        return node_index

    def add_initializer(self, tensor_proto: onnx.TensorProto) -> int:
        """Add an initializer from the model as a node in the DiGraph."""
        if not tensor_proto.name:
            raise ValueError("Can only handled named initializers.")
        name = tensor_proto.name

        self.initializers[name] = tensor_proto
        value_info = onnx.ValueInfoProto()
        value_info.name = name
        value_info.type.tensor_type.elem_type = tensor_proto.data_type
        for dim in tensor_proto.dims:
            value_info.type.tensor_type.shape.dim.append(
                onnx.TensorShapeProto.Dimension(dim_value=dim)
            )
        self.add_value_info(value_info)

        payload = NetworkNode(type=NodeType.INITIALIZER, name=name)
        node_index = self.digraph.add_node(payload)
        self._producer_node_indices[name] = node_index
        self._producer_arg_indices[name] = None

        return node_index

    def add_operator(self, node_proto: onnx.NodeProto) -> int:
        """Add an ONNX node, representing an operator, to the DiGraph."""
        node_payload = NetworkNode(
            type=NodeType.OPERATOR,
            name=node_proto.name,
            onnx_node=node_proto,
        )
        node_index = self.digraph.add_node(node_payload)

        for input_index, input_name in enumerate(node_proto.input):
            if input_name == "":
                continue  # Skipped optional input.

            producer_index = self._producer_node_indices[input_name]
            edge_payload = NetworkEdge(
                name=input_name,
                value_info=self.value_infos[input_name],
                producer_arg_index=self._producer_arg_indices[input_name],
                consumer_arg_index=input_index,
            )
            # Add an edge connecting the producer of this input to the new
            # node.
            self.digraph.add_edge(producer_index, node_index, edge_payload)

        for output_index, output_name in enumerate(node_proto.output):
            self._producer_node_indices[output_name] = node_index
            self._producer_arg_indices[output_name] = output_index

        return node_index

    def add_model_output(self, value_info: onnx.ValueInfoProto) -> int:
        """Adds an output value of an ONNX model to the DiGraph."""
        if not value_info.name:
            raise ValueError("Outputs must be named.")
        name = value_info.name
        self.add_value_info(value_info)

        node_payload = NetworkNode(type=NodeType.OUTPUT, name=name)
        node_index = self.digraph.add_node(node_payload)

        edge_payload = NetworkEdge(
            name=name,
            value_info=value_info,
            producer_arg_index=self._producer_arg_indices[name],
            consumer_arg_index=None,
        )
        self.digraph.add_edge(
            self._producer_node_indices[name],
            node_index,
            edge_payload,
        )

        return node_index

    def _add_model_graph(self, graph: onnx.GraphProto):
        """Add the whole model graph to the DiGraph."""
        if len(graph.sparse_initializer) > 0:
            raise NotImplementedError("Sparse initializers not handled")

        for initializer in graph.initializer:
            self.add_initializer(initializer)

        for input_info in graph.input:
            self.add_model_input(input_info)

        for value_info in graph.value_info:
            self.add_value_info(value_info)

        # Add output info before self.add_operator in case
        # some nodes are both output nodes and input
        # of some intermediate nodes
        for output_info in graph.output:
            self.add_value_info(output_info)

        for node in graph.node:
            self.add_operator(node)

        for output_info in graph.output:
            self.add_model_output(output_info)

    def iter_nodes(self) -> typing.Generator[NetworkNode, None, None]:
        """Iterate over node data in the network.

        Yields the node payload objects, as NetworkNode.
        """
        for node_index in self.digraph.node_indices():
            node_payload = self.digraph.get_node_data(node_index)
            yield node_payload

    def find_node_by_name(self, name: str) -> int:
        """Finds the index of a node with the given name.

        Does a linear search. Note that not all nodes are guaranteed to have a
        name.
        """
        for node_index in self.digraph.node_indices():
            node_payload = self.digraph.get_node_data(node_index)
            if node_payload.name == name:
                return node_index

        raise KeyError(f"No node with name {name}")

    def remove_node(self, node_index: int):
        """Remove a node from the ModelNetwork given its index."""
        value_names = set()
        for edge_index in self.digraph.incident_edges(node_index):
            value_name = self.digraph.get_edge_data_by_index(edge_index).name
            value_names.add(value_name)
        for value_name in value_names:
            assert (
                self._producer_node_indices[value_name] == node_index
            ), f"Unexpected producer index for {value_name}"
            del self._producer_node_indices[value_name]
            del self._producer_arg_indices[value_name]

        node_payload = self.digraph.get_node_data(node_index)
        if node_payload.type == NodeType.INITIALIZER:
            del self.initializers[node_payload.name]

        self.digraph.remove_node(node_index)

    def verify_producer_indices(self):
        """Asserts correct contents of internal book-keeping on values."""
        for node_index in self.digraph.node_indices():
            node_payload = self.digraph.get_node_data(node_index)
            if node_payload.type == NodeType.OPERATOR:
                assert node_payload.onnx_node is not None
                for output_index, output_name in enumerate(
                    node_payload.onnx_node.output
                ):
                    assert output_name in self._producer_node_indices
                    assert self._producer_node_indices[output_name] == node_index
                    assert output_name in self._producer_arg_indices
                    assert self._producer_arg_indices[output_name] == output_index
            elif node_payload.type in [NodeType.INITIALIZER, NodeType.INPUT]:
                assert node_payload.name in self._producer_node_indices
                assert self._producer_node_indices[node_payload.name] == node_index
                assert node_payload.name in self._producer_arg_indices
                assert self._producer_arg_indices[node_payload.name] is None

    def verify_edges(self):
        """Asserts that the graph edges match the contents of the nodes."""
        for edge_data in self.digraph.edge_index_map().values():
            src_node_index, dst_node_index, edge_payload = edge_data

            src_node_payload = self.digraph.get_node_data(src_node_index)
            if src_node_payload.type == NodeType.OPERATOR:
                assert src_node_payload.onnx_node is not None
                assert edge_payload.producer_arg_index is not None
                assert (
                    src_node_payload.onnx_node.output[edge_payload.producer_arg_index]
                    == edge_payload.name
                )
            elif src_node_payload.type in [NodeType.INITIALIZER, NodeType.INPUT]:
                assert edge_payload.producer_arg_index is None
                assert src_node_payload.name == edge_payload.name
            else:
                raise AssertionError(
                    f"Unexpected src node type: {src_node_payload.type}"
                )

            dst_node_payload = self.digraph.get_node_data(dst_node_index)
            if dst_node_payload.type == NodeType.OPERATOR:
                assert dst_node_payload.onnx_node is not None
                assert edge_payload.consumer_arg_index is not None
                assert (
                    dst_node_payload.onnx_node.input[edge_payload.consumer_arg_index]
                    == edge_payload.name
                )
            elif dst_node_payload.type == NodeType.OUTPUT:
                assert edge_payload.consumer_arg_index is None
                assert dst_node_payload.name == edge_payload.name
            else:
                raise AssertionError(
                    f"Unexpected dst node type: {dst_node_payload.type}"
                )

        for node_index in self.digraph.node_indices():
            node_payload = self.digraph.get_node_data(node_index)

            if node_payload.type == NodeType.OPERATOR:
                assert node_payload.onnx_node is not None
                input_names = [
                    input_name
                    for input_name in node_payload.onnx_node.input
                    if input_name
                ]
                output_names = [
                    output_name
                    for output_name in node_payload.onnx_node.output
                    if output_name
                ]
            elif node_payload.type in [NodeType.INPUT, NodeType.INITIALIZER]:
                input_names = []
                output_names = [node_payload.name]
            elif node_payload.type == NodeType.OUTPUT:
                input_names = [node_payload.name]
                output_names = []
            else:
                raise AssertionError(f"Unexpected node type: {node_payload.type}")

            in_edge_names = [
                edge_data.name for _, _, edge_data in self.digraph.in_edges(node_index)
            ]
            out_edge_names = [
                edge_data.name for _, _, edge_data in self.digraph.out_edges(node_index)
            ]
            assert set(input_names) == set(in_edge_names)
            if not set(output_names).issuperset(set(out_edge_names)):
                unexpected_edges = set(out_edge_names).difference(output_names)
                raise AssertionError(f"No outputs for edges: {unexpected_edges}")

    def verify(self):
        """Assert correctness of the model network.

        Checks that e.g. the metadata associated with the nodes & edges matches
        the graph topology.
        """
        self.verify_edges()
        self.verify_producer_indices()

    def find_subgraphs(
        self,
        template: SubgraphTemplate,
        disjoint: bool = False,
    ) -> typing.Generator[MatchedSubgraph, None, None]:
        """Finds subgraphs in the model that match a given template."""

        def template_matcher(obj, template_func):
            if callable(obj):
                obj, template_func = template_func, obj
            return template_func(obj)

        matched_indices: typing.Set[int] = set()
        for match in rx.vf2_mapping(
            self.digraph,
            template.digraph,
            subgraph=True,
            node_matcher=template_matcher,
            edge_matcher=template_matcher,
        ):
            graph_indices = list(map(operator.itemgetter(0), match.items()))
            if disjoint and matched_indices.intersection(graph_indices):
                continue
            matched_indices.update(graph_indices)
            yield MatchedSubgraph(match, self, template)

    def substitute_matched_graph(
        self,
        old_subgraph: MatchedSubgraph,
        new_subgraph: onnx.ModelProto,
        keep_old_nodes: typing.Optional[typing.Sequence[int]] = None,
    ):
        """Edits the graph to replace one subgraph with another.

        The part of the model matched by `old_subgraph` will be deleted, and
        the `new_subgraph` will be added. Links between nodes for input/output
        values that are common between `old_subgraph` and `new_subgraph` will
        be updated.

        If some template indices from the match are included in keep_old_nodes,
        these will not be deleted. Then the graph will both have some nodes
        from the old matched subgraph, and from new_subgraph.
        """
        nodes_to_remove = set(old_subgraph.model_node_indices())

        keep_old_nodes = keep_old_nodes or []
        for template_node_index in keep_old_nodes:
            model_node_index = old_subgraph.reverse_match_dict[template_node_index]
            nodes_to_remove.remove(model_node_index)

        output_names = {
            subgraph_output_info.name
            for subgraph_output_info in new_subgraph.graph.output
        }
        output_consumers = collections.defaultdict(set)
        output_consumer_arg_index = {}
        for node_index in nodes_to_remove:
            for consumer in self.digraph.successor_indices(node_index):
                if consumer in nodes_to_remove:
                    continue
                edge_data = self.digraph.get_edge_data(node_index, consumer)
                if edge_data.name in output_names:
                    output_consumers[edge_data.name].add(consumer)
                    output_consumer_arg_index[edge_data.name, consumer] = (
                        edge_data.consumer_arg_index
                    )

        for old_node_index in nodes_to_remove:
            self.remove_node(old_node_index)

        inferred_model = shape_inference.infer_shapes(new_subgraph)
        for initializer in inferred_model.graph.initializer:
            self.add_initializer(initializer)
        for value_info in inferred_model.graph.value_info:
            self.add_value_info(value_info)
        for node_proto in inferred_model.graph.node:
            op_index = self.add_operator(node_proto)
            for output_arg_index, output_name in enumerate(node_proto.output):
                for consumer in output_consumers[output_name]:
                    edge_payload = NetworkEdge(
                        name=output_name,
                        value_info=self.value_infos[output_name],
                        producer_arg_index=output_arg_index,
                        consumer_arg_index=output_consumer_arg_index[
                            output_name, consumer
                        ],
                    )
                    self.digraph.add_edge(op_index, consumer, edge_payload)

    def as_onnx_model_proto(self) -> onnx.ModelProto:
        """Create an ONNX ModelProto from this network."""
        graph_proto = onnx.GraphProto(name=self.graph_name)
        # Ops must be in a computable order.
        order = rx.topological_sort(self.digraph)
        for node_index in order:
            node_data = self.digraph.get_node_data(node_index)
            if node_data.type == NodeType.INITIALIZER:
                graph_proto.initializer.append(self.initializers[node_data.name])
            elif node_data.type == NodeType.INPUT:
                graph_proto.input.append(self.value_infos[node_data.name])
            elif node_data.type == NodeType.OUTPUT:
                graph_proto.output.append(self.value_infos[node_data.name])
            elif node_data.type == NodeType.OPERATOR:
                graph_proto.node.append(node_data.onnx_node)
            else:
                raise ValueError(f"Unhandled node type: {node_data.type}")

        model = onnx.helper.make_model(
            graph_proto,
            opset_imports=self.opset_import,
            functions=self.functions,
        )
        return model

    def draw(self, filename: os.PathLike):
        """Write a drawing of this network to an image file."""

        def node_attr_fn(node_data):
            if node_data.type == NodeType.OPERATOR:
                return {"shape": "box", "label": node_data.name}
            elif node_data.type == NodeType.INPUT:
                return {"shape": "triangle", "label": ""}
            elif node_data.type == NodeType.OUTPUT:
                return {"shape": "invtriangle", "label": ""}
            elif node_data.type == NodeType.INITIALIZER:
                return {"shape": "circle", "label": ""}

        def edge_attr_fn(edge_data):
            return {"label": edge_data.name}

        rustworkx.visualization.graphviz_draw(
            self.digraph,
            node_attr_fn=node_attr_fn,
            edge_attr_fn=edge_attr_fn,
            filename=os.fspath(filename),
        )

    def convert_initializer_to_scalar(self, initializer_name: str) -> typing.Any:
        """Helper function to convert the initializer to scalar"""
        initializer = self.initializers[initializer_name]
        return onnx.numpy_helper.to_array(initializer).item()

    def convert_initializer_to_np_array(self, initializer_name: str) -> np.ndarray:
        """Helper function to convert the initializer to numpy array"""
        initializer = self.initializers[initializer_name]
        return onnx.numpy_helper.to_array(initializer)
