from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
# @build:deps [
# @/deeplearning/projects/faiss:pyfaiss
# ]

from caffe2.python import model_helper, workspace
from caffe2.python.predictor.predictor_exporter import PredictorExportMeta

from ml.rl.preprocessing.preprocessor_net import PreprocessorNet, MISSING_VALUE
from ml.rl.training.rl_predictor import RLPredictor

import logging
logger = logging.getLogger(__name__)

TRAIN_REWARD_MODEL = -1
from enum import Enum


class QDRRN_MODEL_T(Enum):
    DQN = 1
    DQNA = 2
    DRRN = 3
    KNN_DRRN = 4
    ACTORCRITIC = 5
    KNN_ACTORCRITIC = 6


MODEL_T_DICT = dict([(op.name, op) for op in QDRRN_MODEL_T])
default_qdrrn_model_type = QDRRN_MODEL_T.DQN

Q_LABEL = "Q"
CRITIC_LABEL = "Q"
STATE_LABEL = "State"
ACTION_LABEL = "Action"
FINAL_OUTPUT_LABEL = CRITIC_LABEL

DEFAULT_USE_PRESERVE = True
DECAY_UPDATE = True


class ContinuousActionPredictor(RLPredictor):
    def predict(self, states, actions):
        """ Returns values for each state

        :param states states as feature -> value dict
        :param actions actions as action_feature -> value dict
        """
        previous_workspace = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace(self._workspace_id)
        for input_blob in states:
            workspace.FeedBlob(
                input_blob,
                np.atleast_1d(states[input_blob]).astype(np.float32)
            )
        for input_blob in actions:
            workspace.FeedBlob(
                input_blob,
                np.atleast_1d(actions[input_blob]).astype(np.float32)
            )

        workspace.RunNetOnce(self._net)
        result = {
            output: workspace.FetchBlob(output)
            for output in self._output_blobs
        }
        workspace.SwitchWorkspace(previous_workspace)
        return result

    def get_predictor_export_meta(self):
        return PredictorExportMeta(
            predict_net=self._net,
            parameters=self._parameters + self._input_blobs,
            inputs=[],
            outputs=self._output_blobs
        )

    @classmethod
    def from_trainers(
        cls,
        trainers,
        state_features,
        action_features,
        state_normalization_parameters,
        action_normalization_parameters,
        trainers_label=None
    ):
        """ Creates ContinuousActionPredictor from a list of action trainers

        :param action_trainers list of QdrrnActionTrainer
        :param features list of state feature names
        :param actions list of action names
        """
        # ensure state and action has no intersect ?
        assert (len(set(state_features) & set(action_features)) == 0)

        input_blobs = state_features[:] + action_features[:]
        model = model_helper.ModelHelper(name="predictor")
        net = model.net

        normalized_input_blobs = []
        normalized_state_features_blob = []
        normalized_action_features_blob = []
        normalizer = PreprocessorNet(net)
        parameters = normalizer.parameters[:]
        zero = "ZERO_from_trainers"
        workspace.FeedBlob(zero, np.array(0))
        parameters.append(zero)

        for input_blob in input_blobs:
            workspace.FeedBlob(
                input_blob, MISSING_VALUE * np.ones(1, dtype=np.float32)
            )
            reshaped_input_blob = input_blob + "_reshaped"
            net.Reshape(
                [input_blob],
                [reshaped_input_blob, input_blob + "_original_shape"],
                shape=[-1, 1]
            )
            normalization_parameters = None
            if input_blob in state_normalization_parameters:
                normalization_parameters = state_normalization_parameters
            elif input_blob in action_normalization_parameters:
                normalization_parameters = action_normalization_parameters
            else:
                logger.info("WARNINING: NORMALIZED PARAMETER NOT FOUND")
                continue
            normalized_input_blob, blob_parameters = \
                normalizer.preprocess_blob(reshaped_input_blob,
                                           normalization_parameters[input_blob])
            parameters.extend(blob_parameters)
            normalized_input_blobs.append(normalized_input_blob)
            if input_blob in state_normalization_parameters:
                normalized_state_features_blob.append(normalized_input_blob)
            if input_blob in action_normalization_parameters:
                normalized_action_features_blob.append(normalized_input_blob)

        is_drrn = len(trainers) == 2 and \
            STATE_LABEL in trainers_label and \
            ACTION_LABEL in trainers_label
        is_actorcritic = len(trainers) == 3 and \
            STATE_LABEL in trainers_label and ACTION_LABEL in trainers_label \
            and CRITIC_LABEL in trainers_label

        input_blob = cls.normalized_input  # "PredictorInput"
        # workspace.FeedBlob(input_blob, np.zeros(1, dtype=np.float32))
        output_dim = "PredictorOutputDim"
        # net.Concat(input_blobs, [input_blob, output_dim], axis=1)
        for i, inp in enumerate(normalized_input_blobs):
            logger.info("input# {}: {}".format(i, inp))
        net.Concat(normalized_input_blobs, [input_blob, output_dim], axis=1)

        input_blob_state = STATE_LABEL + input_blob
        input_blob_action = ACTION_LABEL + input_blob
        output_dim_state = STATE_LABEL + output_dim
        output_dim_action = ACTION_LABEL + output_dim

        net.Concat(
            normalized_state_features_blob,
            [input_blob_state, output_dim_state],
            axis=1
        )
        net.Concat(
            normalized_action_features_blob,
            [input_blob_action, output_dim_action],
            axis=1
        )

        input_blob_critic = CRITIC_LABEL + input_blob
        output_dim_critic = CRITIC_LABEL + output_dim
        workspace.FeedBlob(input_blob_critic, np.zeros(1, dtype=np.float32))
        # regular net construction
        output_blobs = []

        for trainer_index, trainer in enumerate(trainers):
            output_name = trainers_label[trainer_index]
            workspace.FeedBlob(output_name, np.zeros(1, dtype=np.float32))
            input_blob_used = input_blob
            if is_drrn or is_actorcritic:
                input_blob_used = output_name + input_blob

            if is_actorcritic and CRITIC_LABEL in output_name:
                output_blob_state = [
                    s for s in trainers_label if STATE_LABEL in s
                ]
                output_blob_state = output_blob_state[0] if len(output_blob_state)\
                    > 0 else STATE_LABEL
                output_blob_action = [
                    s for s in trainers_label if ACTION_LABEL in s
                ]
                output_blob_action = output_blob_action[0] if len(output_blob_action)\
                    > 0 else ACTION_LABEL
                net.Concat(  # [input_blob_state, output_blob_state], #critic=state
                    [input_blob_state, output_blob_action],  # critic=action
                    [input_blob_critic, output_dim_critic], axis=1)

            parameters.extend(
                trainer.build_predictor(model, input_blob_used, output_name)
            )
            output_blobs.append(output_name)

        # for action trainer build the join
        if is_drrn:
            join_output = FINAL_OUTPUT_LABEL
            workspace.FeedBlob(join_output, np.zeros(1, dtype=np.float32))
            for trainer_index, trainer in enumerate(trainers[:2]):
                output_name = trainers_label[trainer_index]
                index_of_otherside = 1 - trainer_index
                trainer.extend_predictor(
                    model, output_name, output_blobs[index_of_otherside],
                    join_output
                )
            # now overwrite the final output blob
            output_blobs += [join_output]
            output_blobs = [FINAL_OUTPUT_LABEL]

        if is_actorcritic:
            join_output = "q_actor" + STATE_LABEL
            # disable the line below if want to return actor output
            output_blobs = [CRITIC_LABEL]
            critic_trainer = trainers[2]._trainer
            for trainer_index, trainer in enumerate(trainers[:2]):
                output_name = trainers_label[trainer_index]
                join_output = "q_actor" + output_name
                workspace.FeedBlob(join_output, np.zeros(1, dtype=np.float32))
                index_of_otherside = 1 - trainer_index
                trainer.extend_predictor(
                    model, output_name, input_blob_state, join_output,
                    critic_trainer
                )
                output_blobs += [join_output]
            # NOTE: here can we use critic output? due to one input is deps
            # on actor to finish, which cannot be enforced. the eval is incorrect
            output_blobs = [FINAL_OUTPUT_LABEL]  # FINAL_OUTPUT_LABEL]

        # for input_blob in input_blobs:
        #     net.ConstantFill([input_blob], [input_blob], value=0.)
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net)

        # from caffe2.python import net_drawer
        # graph = net_drawer.GetPydotGraph(
        #     net.Proto().op, "predictor", rankdir="LR")
        # graph.write_png('_'.join(trainers_label) + '_predictor.png')

        predictor = cls(
            net, input_blobs, output_blobs, parameters,
            workspace.CurrentWorkspace()
        )
        return predictor
