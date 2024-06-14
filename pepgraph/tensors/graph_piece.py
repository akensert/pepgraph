import tensorflow as tf
import typing 

from pepgraph import types



class GraphPieceBatchEncoder(tf.experimental.ExtensionTypeBatchEncoder):

    def batch(
        self, 
        spec: 'GraphPiece.Spec', 
        batch_size: int
    ) -> 'GraphPiece.Spec':
      
        def batch_field(
            f: typing.Union[tf.TypeSpec, typing.Any]
        ) -> typing.Union[tf.TypeSpec, typing.Any]:
            
            if isinstance(f, tf.TensorSpec):
                return tf.TensorSpec(
                    shape=[None] + f.shape[1:],
                    dtype=f.dtype
                )
            elif isinstance(f, tf.RaggedTensorSpec):
                return tf.RaggedTensorSpec(
                    shape=[batch_size] + [None] + f.shape[1:],
                    dtype=f.dtype,
                    ragged_rank=1,
                    row_splits_dtype=f.row_splits_dtype
                )
            elif isinstance(f, tf.TypeSpec):
                return f.__batch_encoder__.batch(f, batch_size)
            return f
                
        fields = tuple(spec.__dict__.items())
        batched_fields = tf.nest.map_structure(batch_field, fields)

        batched_spec = object.__new__(type(spec))
        batched_spec.__dict__.update(batched_fields)
        if batched_spec.__dict__['sizes'] is not None:
            batched_spec.__dict__['sizes'] = tf.TensorSpec(
                shape=[batch_size], dtype=spec.sizes.dtype)

        return batched_spec
    
    def unbatch(
        self, 
        spec: 'GraphPiece.Spec'
    ) -> 'GraphPiece.Spec':
   
        def unbatch_field(
            f: typing.Union[tf.TypeSpec, typing.Any]
        ) -> typing.Union[tf.TypeSpec, typing.Any]:
            
            if isinstance(f, tf.TensorSpec):
                return tf.TensorSpec(
                    shape=[None] + f.shape[1:],
                    dtype=f.dtype
                )
            elif isinstance(f, tf.RaggedTensorSpec):
                return tf.RaggedTensorSpec(
                    shape=[None] + f.shape[2:],
                    dtype=f.dtype,
                    ragged_rank=0,
                    row_splits_dtype=f.row_splits_dtype
                )
            elif isinstance(f, tf.TypeSpec):
                return f.__batch_encoder__.unbatch(f)
            return f
        
        fields = tuple(spec.__dict__.items())
        unbatched_fields = tf.nest.map_structure(unbatch_field, fields)
        
        unbatched_spec = object.__new__(type(spec))
        unbatched_spec.__dict__.update(unbatched_fields)
        if unbatched_spec.__dict__['sizes'] is not None:
            unbatched_spec.__dict__['sizes'] = tf.TensorSpec(
                shape=[], dtype=spec.sizes.dtype)

        return unbatched_spec
        
    def encode(
        self, 
        spec: 'GraphPiece.Spec', 
        value: 'GraphPiece', 
        minimum_rank: int = 0
    ) -> typing.Tuple[types.TensorOrCompositeTensor, ...]:
        
        if _flattened_state(spec):
            value = value.unflatten(_recursive=False)
        
        value_components = tuple(value.__dict__[key] for key in spec.__dict__)

        value_components = tuple(
            x for x in tf.nest.flatten(value_components) 
            if isinstance(x, types.TensorOrCompositeTensor)
        )
        return value_components
    
    def encoding_specs(
        self, 
        spec: 'GraphPiece.Spec', 
    ) -> typing.Tuple[tf.TypeSpec, ...]:
        
        def encode_fields(
            f: typing.Union[tf.TypeSpec, typing.Any]
        ) -> typing.Union[tf.TypeSpec, typing.Any]:
            
            if isinstance(f, tf.TensorSpec) and spec.sizes is not None:
                scalar = spec.sizes.shape.rank == 0
                return tf.RaggedTensorSpec(
                    shape=([None] if scalar else [None, None]) + f.shape[1:], 
                    dtype=f.dtype, 
                    ragged_rank=(0 if scalar else 1),
                    row_splits_dtype=spec.sizes.dtype)
            return f
        
        encoded_fields = tf.nest.map_structure(encode_fields, spec.__dict__)
        encoded_fields['sizes'] = spec.sizes

        spec_components = tuple(encoded_fields.values())
        
        spec_components = tuple(
            x for x in tf.nest.flatten(spec_components) 
            if isinstance(x, tf.TypeSpec)
        )

        return spec_components
    
    def decode(
        self, 
        spec: 'GraphPiece.Spec', 
        encoded_value: typing.Tuple[types.TensorOrCompositeTensor, ...]
    ) -> 'GraphPiece':
        
        spec_tuple = tuple(spec.__dict__.values())
        
        encoded_value = iter(encoded_value)

        value_tuple = [
            next(encoded_value) if isinstance(x, tf.TypeSpec) else x
            for x in tf.nest.flatten(spec_tuple)
        ]
                
        value_tuple = tf.nest.pack_sequence_as(spec_tuple, value_tuple)
        fields = dict(zip(spec.__dict__.keys(), value_tuple))
        
        value = object.__new__(spec.value_type)
        value.__dict__.update(fields)

        if _flattened_state(spec):
            return value.flatten(_recursive=False)
        return value


class GraphPiece(tf.experimental.BatchableExtensionType):
    
    __name__ = 'GraphPiece'

    sizes: typing.Optional[tf.Tensor] = None
    
    __batch_encoder__ = GraphPieceBatchEncoder()

    def __validate__(self):
        if not tf.executing_eagerly():
            return
        data = self.data 
        sizes = data.pop('sizes', None)
        if sizes is None:
            return
        size = tf.reduce_sum(sizes)
        def validate(x):
            if isinstance(x, tf.Tensor):
                assert tf.equal(size, tf.shape(x, out_type=size.dtype)[0])
            elif isinstance(x, tf.RaggedTensor):
                assert tf.equal(tf.shape(sizes)[0], tf.shape(x)[0])
            assert True
        tf.nest.map_structure(validate, data)

    def __getitem__(self, index):
        # TODO: Edge sets should not only be indexed from a GraphTensor
        #       instance. Raise error or warning otherwise? 
        def get_item(x):
            if isinstance(x, GraphPiece):
                return x.__getitem__(index)
            elif isinstance(x, tf.RaggedTensor):
                return x[index]
            elif isinstance(x, types.TensorOrCompositeTensor):
                return tf.nest.map_structure(
                    lambda x_i: x_i[index], x, expand_composites=True)
            return x
        if _flattened_state(self):
            data = self.unflatten(_recursive=False).data 
        else:
            data = self.data
        data = tf.nest.map_structure(get_item, data)
        tensor = self.__class__(**data)
        if _flattened_state(self):
            return tensor.flatten(_recursive=False)
        return tensor 

    def flatten(self, _recursive: bool = True):
        if _flattened_state(self) and _recursive:
            class_name = self.__class__.__qualname__
            raise ValueError(
                f'{class_name} instance already in flattened state.'
            )
        def flatten_value(x):
            if isinstance(x, GraphPiece) and _recursive:
                return x.flatten(_recursive=_recursive)
            if isinstance(x, tf.RaggedTensor):
                return x.flat_values
            return x
        return self.__class__(
            **tf.nest.map_structure(flatten_value, self.data)
        )

    def unflatten(self, _recursive: bool = True):
        if _unflattened_state(self) and _recursive:
            class_name = self.__class__.__qualname__
            raise ValueError(
                f'{class_name} instance already in unflattened state.'
            )
        scalar = self.sizes is None or self.sizes.shape.rank == 0
        def unflatten_value(x):
            if isinstance(x, GraphPiece) and _recursive:
                return x.unflatten(_recursive=_recursive)
            if isinstance(x, tf.Tensor) and not scalar:
                return tf.RaggedTensor.from_row_lengths(x, self.sizes)
            return x
        data = self.data
        sizes = data.pop('sizes')
        data = tf.nest.map_structure(unflatten_value, data)
        data['sizes'] = sizes
        return self.__class__(**data)
    
    def update(self, **kwargs):
        data = self.data
        for (key, value) in kwargs.items():
            if isinstance(value, typing.Mapping):
                data[key].update(value)
            else:
                data[key] = value
        return self.__class__(**data)
        
    @property
    def shape(self):
        return tf.TensorShape(self.sizes.shape + [None])
    
    @property
    def dtype(self):
        return self.sizes.dtype
    
    @property 
    def rank(self):
        return self.shape.rank
    
    @property
    def spec(self):
        spec = tf.type_spec_from_value(self)
        if spec.sizes is None:
            def f(x):
                if isinstance(x, GraphPiece):
                    return x.spec
                elif isinstance(x, types.TensorOrCompositeTensor):
                    return tf.type_spec_from_value(x)
                return x
            return spec.__class__(**tf.nest.map_structure(f, self.data))
        elif spec.sizes.shape.rank == 0:
            return spec._batch(None)._unbatch()
        return spec._unbatch()._batch(None)

    @property
    def data(self):
        data = {}
        for key in tf.type_spec_from_value(self).__dict__:
            value = self.__dict__[key]
            if isinstance(value, typing.Mapping):
                value = dict(value)
            data[key] = value
        return data
    

def _specifies_ragged_tensor(spec):
    def f(x):
        if isinstance(x, (tf.TensorSpec, tf.RaggedTensorSpec)):
            return isinstance(x, tf.RaggedTensorSpec) and x.ragged_rank > 0
        elif isinstance(x, tf.TypeSpec):
            return _specifies_ragged_tensor(x)
        return False
    return tf.nest.flatten(tf.nest.map_structure(f, spec.__dict__))
    
def _unflattened_state(spec):
    if isinstance(spec, types.TensorOrCompositeTensor):
        spec = tf.type_spec_from_value(spec)
    return any(_specifies_ragged_tensor(spec))

def _flattened_state(spec):
    if isinstance(spec, types.TensorOrCompositeTensor):
        spec = tf.type_spec_from_value(spec)
    return not any(_specifies_ragged_tensor(spec))