import tensorflow as tf
import typing 

from pepgraph import types 

from pepgraph.tensors.graph_piece import GraphPiece



class NodeSet(GraphPiece):
    
    __name__ = 'NodeSet'

    features: typing.Optional[types.TensorOrRaggedTensor] = None
        

class EdgeSet(GraphPiece):
    
    __name__ = 'EdgeSet'

    source: typing.Tuple[str, types.TensorOrRaggedTensor]
    target: typing.Tuple[str, types.TensorOrRaggedTensor]
    features: typing.Optional[types.TensorOrRaggedTensor] = None


class Context(tf.experimental.BatchableExtensionType):

    __name__ =  'Context'

    data: typing.Mapping[str, tf.Tensor]
    
    
class GraphTensor(GraphPiece):

    __name__ = 'GraphTensor'

    node_sets: typing.Mapping[str, NodeSet]
    edge_sets: typing.Mapping[str, EdgeSet]
    context: typing.Optional[Context] = None

    def flatten(self, *, _recursive: bool = True):
        tensor = super().flatten(_recursive=_recursive)
        return _adapt_node_indices(tensor, mode='add') 
    
    def unflatten(self, *, _recursive: bool = True):
        tensor = super().unflatten(_recursive=_recursive)
        return _adapt_node_indices(tensor, mode='subtract') 

    @property
    def shape(self):
        node_set = tf.nest.flatten(self.node_sets)[0]
        return tf.TensorShape(node_set.sizes.shape[:1] + [None])

    @property
    def dtype(self):
        node_set = tf.nest.flatten(self.node_sets)[0]
        return node_set.sizes.dtype
    
    class Spec:
        
        @property
        def shape(self):
            node_set = tf.nest.flatten(self.node_sets)[0]
            return tf.TensorShape(node_set.sizes.shape[:1] + [None])
        
        @property
        def dtype(self):
            node_set = tf.nest.flatten(self.node_sets)[0]
            return node_set.sizes.dtype
        

def _adapt_node_indices(
    tensor: 'GraphTensor', 
    mode: str
) -> 'GraphTensor':

    func = getattr(tf.math, mode)

    adapted_edge_sets = {}

    for (name, edge_set) in tensor.edge_sets.items():
        if edge_set.sizes.shape.rank == 0:
            continue

        source_tag, source_indices = edge_set.source 
        target_tag, target_indices = edge_set.target 
        source_sizes = tensor.node_sets[source_tag].sizes
        target_sizes = tensor.node_sets[target_tag].sizes

        source_row_starts = tf.cumsum(tf.concat([[0], source_sizes[:-1]], axis=0))
        target_row_starts = tf.cumsum(tf.concat([[0], target_sizes[:-1]], axis=0))
        source_row_starts = tf.cast(source_row_starts, source_indices.dtype)
        target_row_starts = tf.cast(target_row_starts, target_indices.dtype)

        if isinstance(source_indices, tf.RaggedTensor):
            source_row_starts = tf.expand_dims(source_row_starts, 1)
            target_row_starts = tf.expand_dims(target_row_starts, 1)
        else:
            indicator = tf.repeat(
                tf.range(tf.shape(edge_set.sizes)[0]), edge_set.sizes)
            source_row_starts = tf.gather(source_row_starts, indicator)
            target_row_starts = tf.gather(target_row_starts, indicator)
            
        adapted_edge_sets[name] = edge_set.update(
            source=(source_tag, func(source_indices, source_row_starts)),
            target=(target_tag, func(target_indices, target_row_starts))
        )
        
    return tensor.update(edge_sets=adapted_edge_sets)    