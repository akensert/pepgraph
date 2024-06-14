import os
import multiprocessing
import math
import glob
import time
import typing 

import tensorflow as tf

from pepgraph.tensors.graph_tensor import GraphTensor 


def write(
    data: typing.List[typing.Union[str, GraphTensor]], 
    path: str, 
    encoder: typing.Optional[typing.Any]  = None, 
    overwrite: bool = True, 
    num_files: typing.Optional[int] = None, 
    num_processes: typing.Optional[int] = None
) -> None:
    
    if os.path.isdir(path) and not overwrite:
        return 
    
    os.makedirs(path, exist_ok=True)

    spec_filepath = os.path.join(path, 'spec.proto')
    example = data[0]
    if not isinstance(example, GraphTensor):
        example = encoder(example)
    proto = example.spec.experimental_as_proto()
    with open(spec_filepath, 'wb') as fh:
        fh.write(proto.SerializeToString())

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    if num_files is None:
        num_files = min(len(data), num_processes)
        
    chunk_size = math.ceil(len(data) / num_files)
    
    paths = [
        os.path.join(path, f'tfrecord-{i:04d}.tfrecord')
        for i in range(num_files)
    ]
    
    data = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_files)
    ]
    
    processes = []
    
    for path, data_subset in zip(paths, data):
    
        while len(processes) >= num_processes:
            for process in processes:
                if not process.is_alive():
                    processes.remove(process)
            else:
                time.sleep(0.1)
                continue
                
        process = multiprocessing.Process(
            target=_write_tfrecord,
            args=(data_subset, path, encoder)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()         
        

def load(
    path: str, 
    shuffle_tf_records: bool = False
) -> tf.data.Dataset:
    
    with open(os.path.join(path, 'spec.proto'), 'rb') as fh:
        serialized_proto = fh.read()
        
    spec = GraphTensor.Spec.experimental_from_proto(
        GraphTensor.Spec.experimental_type_proto().FromString(serialized_proto)
    )
    
    filenames = sorted(glob.glob(os.path.join(path, '*.tfrecord')))
    num_files = len(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_tf_records:
        ds = ds.shuffle(num_files)
    ds = ds.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=1)
    
    ds = ds.map(
        lambda x: _parse_example(x, spec),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds


def _write_tfrecord(
    data: typing.List[typing.Union[str, GraphTensor]], 
    path: str, 
    encoder: typing.Optional[typing.Any] = None
) -> None:
    
    with tf.io.TFRecordWriter(path) as writer:

        for example in data:
            
            if isinstance(example, GraphTensor):
                composite_tensor = example
            else:
                composite_tensor = encoder(example)
                assert isinstance(composite_tensor, GraphTensor), (
                    'encoder needs to produce GraphTensor instances.')
            
            flat_values = tf.nest.flatten(composite_tensor, expand_composites=True)
            flat_values = [tf.io.serialize_tensor(value).numpy() for value in flat_values]
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=flat_values))
            serialized_feature = _serialize_example({'feature': feature})
            writer.write(serialized_feature)

def _serialize_example(
    feature: dict[str, tf.train.Feature]
) -> bytes:
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _parse_example(
    x: tf.Tensor, 
    spec: GraphTensor.Spec
) -> tf.Tensor:
    out = tf.io.parse_single_example(
        x, features={'feature': tf.io.RaggedFeature(tf.string)})['feature']
    out = [tf.ensure_shape(tf.io.parse_tensor(x[0], s.dtype), s.shape) for (x, s) in zip(
        tf.split(out, len(tf.nest.flatten(spec, expand_composites=True))), 
        tf.nest.flatten(spec, expand_composites=True))]
    out = tf.nest.pack_sequence_as(spec, tf.nest.flatten(out), expand_composites=True)
    return out