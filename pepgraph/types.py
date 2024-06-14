import tensorflow as tf
from tensorflow.python.framework import composite_tensor

import typing


Tensor = tf.Tensor 
RaggedTensor = tf.RaggedTensor
CompositeTensor = composite_tensor.CompositeTensor
TensorOrRaggedTensor = typing.Union[Tensor, RaggedTensor]
TensorOrCompositeTensor = typing.Union[Tensor, CompositeTensor]