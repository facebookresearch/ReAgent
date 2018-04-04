#!/usr/bin/env python3


from ml.rl.thrift.core.ttypes import ContinuousActionModelParameters


class DDPGPredictor(object):
    def __init__(self, parameters: ContinuousActionModelParameters) -> None:
        return None

    def export(self):
        parameters = None
        return DDPGPredictor(parameters)
