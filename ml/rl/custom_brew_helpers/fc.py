from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags


def fc_explicit_param_names(
    model, blob_in, blob_out, weight_init=None, bias_init=None, **kwargs
):
    """
    Sets up operators for a fully connected layer between `blob_in`
    and `blob_out`. Requires that explicit weight and bias parameters
    are supplied.

    :param model: The ModelHelper object whose net the operators
        should be added to
    :blob_in: The input blob for the fully connected layer
    :blob_out: The output blob from the fully connected layer
    :weight_init: Tuple specifying weight initialization information. Its first
        entry is the name of the caffe2 operator to use in creating the weight param
        and second is a dictionary of kwargs it should be passed.
    :bias_init: Tuple specifying bias initialization information. Its first
        entry is the name of the caffe2 operator to use in creating the bias param
        and second is a dictionary of kwargs it should be passed.
    :dim_in: Number of nodes in input layer
    :dim_out: Number of nodes in output layer
    :weight_name: Name of blob corresponding to an initialized weight parameter
    :bias_name: Name of blob corresponding to an initialized bias parameter
    """
    required_kwargs = ['dim_in', 'dim_out', 'weight_name', 'bias_name']
    for arg in required_kwargs:
        assert arg in kwargs, "Please supply kwarg {}".format(arg)
    dim_in, dim_out = kwargs['dim_in'], kwargs['dim_out']

    WeightInitializer = initializers.update_initializer(
        None, weight_init, ("XavierFill", {})
    )
    BiasInitializer = initializers.update_initializer(
        None, bias_init, ("ConstantFill", {})
    )

    weight = model.create_param(
        param_name=kwargs['weight_name'],
        shape=[dim_out, dim_in],
        initializer=WeightInitializer,
        tags=ParameterTags.WEIGHT)
    bias = model.create_param(
        param_name=kwargs['bias_name'],
        shape=[dim_out, ],
        initializer=BiasInitializer,
        tags=[ParameterTags.BIAS])

    return model.net.FC([blob_in, weight, bias], blob_out)
