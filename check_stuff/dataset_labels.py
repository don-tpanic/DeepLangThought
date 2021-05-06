import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.data.ops import dataset_ops


class_labels = np.array([1,2,3], dtype=np.float32)
re = array_ops.one_hot(class_labels, 10)
print(re)

# class_berts = np.random.random((3,2))

# labels = [class_labels, class_berts]

# label_ds = dataset_ops.Dataset.from_tensor_slices(labels)
# label_ds = label_ds.map(lambda x: array_ops.one_hot(x, 10))

# for i in label_ds.as_numpy_iterator():
#     print(i)