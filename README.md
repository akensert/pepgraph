<img src="https://github.com/akensert/pepgraph/blob/main/docs/_static/pepgraph-logo-pixel.png" alt="pepgraph-title" width="90%">

**Work in progress**: Inspired by TF-GNN, this project aims to implement Graph Tensors and Graph Neural Networks using TF's ExtensionType API. While focusing on molecular structures such as peptides, PepGraph can be used for other graph data as well. 

> As Keras 3 does not currently support extension types, this project currently requires Keras 2 (and TF<=2.15)

## Models 

*in progress*

## Graph Tensor 

*in progress*

Obtain a `GraphTensor` instance encoding multiple peptides as a single disjoint graph. In addition to atoms and bonds, virtual nodes are added which corresponds to the residues (amino acids) of the peptides. Relevant atoms are linked to these virtual nodes in a unidirectional way; the features of the virtual nodes can subsequently be extracted for sequence modeling (using e.g., an LSTM).

> Current modules are experimental and may change in the future.

```python 
from pepgraph import GraphTensor, Context, NodeSet, EdgeSet

peptide_graph = GraphTensor(
    context=Context({
        "n_residues": [1, 2, 1]
    }),
    node_sets={        
        "atoms": NodeSet(
            sizes=[5, 9, 6], 
            features=[
                "N", "C", "C", "O", "O",
                "N", "C", "C", "O", "N", "C", "C", "O", "O",
                "N", "C", "C", "C", "O", "O"
            ]
        ),   
        "residues": NodeSet(
            sizes=[1, 2, 1], 
            features=[
                "Gly", 
                "Gly", "Gly", 
                "Ala"
            ]
        ),          
    },
    edge_sets={
        "bonds": EdgeSet(
            sizes=[8, 16, 10], 
            source=(
                "atoms", [
                    0,  1,  1,  2,  2,  2,  3,  4,  
                    5,  6,  6,  7,  7,  7,  8,  9,  9, 10, 10, 11, 11, 11, 12, 13, 
                    14, 15, 15, 15, 16, 17, 17, 17, 18, 19
                ]
            ),
            target=(
                "atoms", [
                    1,  0,  2,  1,  3,  4,  2,  2,  
                    6,  5,  7,  6,  8,  9,  7,  7, 10, 9, 11, 10, 12, 13, 11, 11, 
                    15, 14, 16, 17, 15, 15, 18, 19, 17, 17
                ]
            )
        ),
        "virtual_bonds": EdgeSet(
            sizes=[5, 9, 6], 
            source=(
                "atoms", [
                    0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19,
                ]
            ),
            target=(
                "residues", [
                    0, 0, 0, 0, 0,
                    1, 1, 1, 1, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3,
                ]
            )
        ),
    }
)
```

Although this `GraphTensor` instance contains nested structures of variable sizes, it can be used with TensorFlow's Dataset API, and thus efficiently iterated over for modeling:

```python
ds = tf.data.Dataset.from_tensor_slices(peptide_graph)
ds = ds.shuffle(3)
ds = ds.batch(2)

for x in ds:
    print(x.node_sets["atoms"].features)
    print(x.edge_sets["bonds"].source[1])
```

Save graphs to disk (via `tf.io.TFRecordWriter`):
```python
from pepgraph.datasets import records

peptide_graphs = [peptide_graph[i] for i in range(3)]
records.write(peptide_graphs, "/tmp/tf_records/")
```

Load graphs from disk (via `tf.data.TFRecordDataset`):
```python
ds = records.load("/tmp/tf_records/")
for x in ds.shuffle(3).batch(2):
    pass
```