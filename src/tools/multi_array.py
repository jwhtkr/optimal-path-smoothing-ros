"""Contains functions to convert ROS messages to numpy arrays."""

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import numpy as np

def shape_to_array_dimensions(shape):
    """
    Convert a numpy array shape to ROS array dimensions.

    Parameters
    ----------
    shape : numpy.array.shape
        The numpy array shape to convert.

    Returns
    -------
    MultiArrayDimension
        ROS array dimensions.
    """
    return [
        MultiArrayDimension(
            label=str(i),
            size=x,
            stride=np.prod(shape[i:])
        ) for i, x in enumerate(shape)
    ]

def array_to_multi_array(array):
    """
    Convert a numpy array to ROS multi array.

    Parameters
    ----------
    array : numpy.array
        The numpy array to convert.

    Returns
    -------
    Float32MultiArray
        ROS multi array.
    """
    array = np.asarray(array)   
    return Float32MultiArray(
        data=array.flatten(),
        layout=MultiArrayLayout(
            dim=shape_to_array_dimensions(array.shape),
            data_offset=0
        )
    )

def multi_array_to_array(multi_array):
    """
    Convert a ROS multi array to numpy array.

    Parameters
    ----------
    multi_array : Float32MultiArray
        The ROS multi array to convert.

    Returns
    -------
    numpy.array
        Numpy array.
    """
    data_offset = multi_array.layout.data_offset
    dim = multi_array.layout.dim
    reshape_dim = tuple(map(lambda x: x.size, dim))
    array = np.array(multi_array.data[data_offset:])
    return array.reshape(reshape_dim)
