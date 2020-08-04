from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import numpy as np

def shape_to_array_dimensions(shape):
    return [
        MultiArrayDimension(
            label=str(i),
            size=x,
            stride=np.prod(shape[i:])
        ) for i,x in enumerate(shape)
    ]

def array_to_multi_array(array):
    return Float32MultiArray(
        data=array.flatten(),
        layout=MultiArrayLayout(
            dim=shape_to_array_dimensions(array.shape),
            data_offset=0
        )
    )

def multi_array_to_array(multi_array):
    data_offset = multi_array.layout.data_offset
    dim = multi_array.layout.dim
    reshape_dim = tuple(map(lambda x: x.size, dim))
    array = np.array(multi_array.data[data_offset:])
    return array.reshape(reshape_dim)
