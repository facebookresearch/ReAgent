from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import six


def read_data():
    dicts = [{u'186': 0.0, u'74': 0.13486135005951,
              u'124': 1.0, u'179': 0.0, u'413': 473.763927022},
             {u'186': 0.0, u'74': 0.072726093232632,
              u'124': 1.0, u'179': 1.0, u'413': 65.0},
             {u'186': 0.0, u'74': 0.025105571374297,
              u'124': 1.0, u'179': 0.0, u'413': 65.0},
             {u'124': -0.071428575, u'74': 0.59101903438568,
              u'186': 11.0, u'179': 1.0, u'413': 1.0},
             {u'186': 1.0, u'74': 0.39461541175842, u'124': -
              0.071428575, u'179': 0.0, u'413': 50.0},
             {u'186': 1.0, u'74': 0.39461541175842, u'124': -
              0.071428575, u'179': 0.0, u'413': 50.0},
             {u'186': 7.0, u'74': 0.067236438393593,
              u'124': 0.9285714, u'179': 0.0, u'413': 2.0},
             {u'186': 7.0, u'74': 0.067236438393593,
              u'124': 0.9285714, u'179': 0.0, u'413': 2.0},
             {u'124': -0.14285715, u'74': 0.080006204545498,
              u'186': 14.0, u'179': 1.0, u'413': 1.0},
             {u'186': 0.0, u'74': 0.29060563445091,
              u'124': -0.071428575, u'179': 1.0, u'413': 23.0}]
    feature_value_map = {}
    for d in dicts:
        for k, v in six.iteritems(d):
            if k not in feature_value_map:
                feature_value_map[k] = []
            feature_value_map[k].append(v)
    for k in feature_value_map:
        feature_value_map[k] = np.array(feature_value_map[k])
    return feature_value_map
