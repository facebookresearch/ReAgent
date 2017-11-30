from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
import six

# @build:deps [
# @/caffe2/caffe2/fb:log_file_db
# @/caffe2/caffe2/python:caffe2_py
# @/deeplearning/projects/faiss:pyfaiss
# ]
# if need to include faiss_gpu:
# # @/deeplearning/projects/faiss:pyfaiss_gpu

from caffe2.python import workspace

from ml.rl.training.rl_trainer import RLTrainer, replace_nans
from ml.rl.thrift.core.ttypes import \
    DiscreteActionModelParameters
from ml.rl.training.two_tower_trainer import \
    TwoTowerActionTrainer, TwoTowerStateTrainer, \
    Q_LABEL, CRITIC_LABEL, STATE_LABEL, ACTION_LABEL, FINAL_OUTPUT_LABEL
from ml.rl.training.continuous_action_predictor import \
    ContinuousActionPredictor
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.target_network import \
    TargetNetwork

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


MODEL_T_DICT = {
    'DQN': QDRRN_MODEL_T.DQN,
    'DQNA': QDRRN_MODEL_T.DQNA,
    'DRRN': QDRRN_MODEL_T.DRRN,
    'KNN_DRRN': QDRRN_MODEL_T.KNN_DRRN,
    'ACTORCRITIC': QDRRN_MODEL_T.ACTORCRITIC,
    'KNN_ACTORCRITIC': QDRRN_MODEL_T.KNN_ACTORCRITIC,
}
default_qdrrn_model_type = QDRRN_MODEL_T.DQN


class ContinuousActionDQNTrainer(object):
    """ This class implements TD training to train a network for continuous
     state + actions as input, and Q value as output
    """

    def __init__(self, name, parameters, num_action_features, as_Critic=False):
        """

        :param parameters training and network paramteres (parameters)
        """
        self._general_setup(parameters)
        self._overall_trainer_setup(name, parameters)
        self.num_action_features = num_action_features
        self._as_critic = as_Critic

    def _general_setup(self, parameters):
        self._trainer_label = "_" + type(self).__name__
        self._reward_burnin = parameters.rl.reward_burnin
        self._maxq_learning = parameters.rl.maxq_learning
        self._gamma = parameters.rl.gamma
        self.all_possible_actions = None
        self.max_possible_count = -1

    def set_all_possible_actions(self, all_possible_actions):
        if self.all_possible_actions is not None or all_possible_actions is None:
            return
        self.all_possible_actions = all_possible_actions
        self.max_possible_count = len(all_possible_actions)

    def _overall_trainer_setup(self, name, parameters):

        self._trainer = MLTrainer(
            name + self._trainer_label, parameters.training
        )

        self._target_network = TargetNetwork(
            self._trainer, parameters.rl.target_update_rate
        )

    def _get_state_trainer(self, all_trainers):
        found_state_trainer = [
            trainer
            for trainer in all_trainers if STATE_LABEL in trainer._trainer_label
        ]
        found_state_trainer = found_state_trainer[0]
        return found_state_trainer

    def _get_state_trainer_eval(self, all_trainers, states):
        return self._get_state_trainer(all_trainers)._target_network.\
            target_values(states)

    def _get_action_trainer(self, all_trainers):
        found_action_trainer = [
            trainer for trainer in all_trainers
            if ACTION_LABEL in trainer._trainer_label
        ]
        found_action_trainer = found_action_trainer[0]
        return found_action_trainer

    def _get_action_trainer_eval(self, all_trainers, actions):
        return self._get_action_trainer(all_trainers)._target_network.\
            target_values(actions)

    def _inputs(self, states=None, actions=None):
        return np.concatenate([states, actions], axis=1)

    def _predicts(self, states=None, actions=None):
        return self._target_network.target_values(self._inputs(states, actions))

    def _labels(
        self, all_trainers, rewards, next_states, next_actions, is_terminal,
        possible_next_actions, iteration
    ):
        next_actions_used = next_actions if not self._as_critic else\
            self._get_action_trainer_eval(all_trainers, next_actions)

        labels = np.copy(rewards)
        if iteration >= self._reward_burnin and\
                self._reward_burnin != TRAIN_REWARD_MODEL:
            if self._maxq_learning is False:
                all_next_value = self._predicts(next_states,
                                                next_actions_used)[:, 0]
                labels += np.where(is_terminal, 0, self._gamma * all_next_value)
            else:
                total_size = 0
                sizes = []
                for i in range(len(rewards)):
                    if is_terminal[i]:
                        continue
                    size = possible_next_actions[i].shape[0]
                    sizes.append(size)
                    total_size += size
                states_to_score = np.zeros(
                    [total_size, next_states[0].shape[0]], dtype=np.float32
                )
                actions_to_score = np.zeros(
                    [total_size, next_actions[0].shape[0]], dtype=np.float32
                )
                cursor = 0
                for i in range(len(rewards)):
                    if is_terminal[i]:
                        continue
                    possible_actions = possible_next_actions[i]
                    size = possible_actions.shape[0]
                    states_to_score[cursor:(cursor + size)] = np.repeat(
                        next_states[i].reshape(1, next_states[i].shape[0]),
                        possible_actions.shape[0],
                        axis=0
                    )
                    actions_to_score[cursor:(cursor + size)] = possible_actions
                    cursor += size
                next_values = self._predicts(states_to_score, actions_to_score)
                cursor = 0
                for i in range(len(rewards)):
                    if is_terminal[i]:
                        continue
                    size = possible_next_actions[i].shape[0]
                    next_value = np.max(next_values[cursor:(cursor + size)])
                    labels[i] += self._gamma * next_value
                    cursor += size

        return labels[:, np.newaxis]

    def train_batch(
        self, all_trainers, states, actions, rewards, next_states, next_actions,
        is_terminal, possible_next_actions, iteration
    ):
        """ Performs TD updates on one batch

        :param all_trainers DiscreteSingleActionTrainers for all actions
        """
        if iteration == self._reward_burnin:
            self._target_network.enable_slow_updates()
        labels = self._labels(
            all_trainers, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, iteration
        )
        actions_used = actions if not self._as_critic else\
            self._get_action_trainer_eval(all_trainers, actions)

        evaluation = self._trainer.train_batch(
            self._inputs(states, actions_used), labels, evaluate=True
        )
        self._target_network.target_update()
        return evaluation

    def build_predictor(self, *args, **kwargs):
        """ See MLTrainer.build_predictor
        """
        return self._trainer.build_predictor(*args, **kwargs)

    @property
    def loss(self):
        return self._trainer.loss


class ContinuousActionTrainer(RLTrainer):
    """ This class implements TD training for taking input action
    """

    def __init__(
        self,
        state_normalization_parameters,
        action_normalization_parameters,
        parameters,
        all_possible_actions=None
    ):
        """

        :param state_normalization_parameters Dict of state feature
                names to NormalizationParameter
        :param action_normalization_parameters Dict of action feature
                names to NormalizationParameter
        :param parameters training and network paramteres (SARSA_PARAMETERS)
        """
        self.qdrrn_model_type = default_qdrrn_model_type
        self.qdrrn_model_knn_k = None
        self.qdrrn_model_knn_freq = None
        self.qdrrn_model_knn_dynreindex = False
        self.qdrrn_model_knn_dynreindex_threshold = 0.5
        self.qdrrn_model_knn_dynreindex_rand_other = 3

        logger.info("QDRRN_PARAMETER" + str(parameters))
        qdrrnparam = parameters.knn
        if qdrrnparam is not None:
            modeltype_str = str(qdrrnparam.model_type)
            if modeltype_str in MODEL_T_DICT:
                self.qdrrn_model_type = MODEL_T_DICT[modeltype_str]
            else:
                raise ValueError("Invalid model type", modeltype_str)
            model_knn_k = qdrrnparam.knn_k
            if model_knn_k is not None:
                self.qdrrn_model_knn_k = model_knn_k
            model_knn_freq = qdrrnparam.knn_frequency
            if model_knn_freq is not None:
                self.qdrrn_model_knn_freq = model_knn_freq
            model_knn_dynreindex = qdrrnparam.knn_dynreindex
            if model_knn_dynreindex is not None:
                self.qdrrn_model_knn_dynreindex = model_knn_dynreindex
            model_knn_dynreindex_thre = qdrrnparam.knn_dynreindex_threshold
            if model_knn_dynreindex_thre is not None:
                self.qdrrn_model_knn_dynreindex_threshold = model_knn_dynreindex_thre
            model_knn_rand_other = qdrrnparam.knn_dynreindex_rand_other
            if model_knn_rand_other is not None:
                self.qdrrn_model_knn_dynreindex_rand_other = model_knn_rand_other

        if all_possible_actions is not None:
            default_none_actions = all_possible_actions
            logger.info(
                "Default all possible actions directly assigned: shape" +
                str(default_none_actions.shape)
            )
        else:
            default_none_action_blob = "default_none_actions"
            default_none_actions = workspace.FetchBlob(default_none_action_blob) \
                if workspace.HasBlob(default_none_action_blob) else None
            if default_none_actions is not None:
                logger.info(
                    "Default all possible actions assigned from workspace: " +
                    str(default_none_actions.shape)
                )
            else:
                logger.info("WARNING: INITIALZED NONE ACTION BY DEFAULT!")
                default_none_actions = np.eye(
                    len(action_normalization_parameters), dtype=np.float32
                )

        logger.info("Model Type " + str(self.qdrrn_model_type.name))
        self._setup_trainer(
            parameters,
            len(state_normalization_parameters),
            len(action_normalization_parameters)
        )

        for trainer in self._trainers:
            trainer.set_all_possible_actions(default_none_actions)

        self._discount_factor = parameters.rl.gamma
        self.minibatch_size = parameters.training.minibatch_size

        self._iteration = 0
        self._state_features = list(state_normalization_parameters.keys())

        # base_from_state = len(self._state_features)
        self._actions = list(action_normalization_parameters.keys())
        self._state_normalization_parameters = state_normalization_parameters
        self._action_normalization_parameters = action_normalization_parameters

    def _prepare_actions(self, actions):
        """
        Replaces NaNs in input `actions `with 0.0. This should be used on
        parametric actions in dense matrix form.

        :param actions: Numpy array with shape (batch_size, action_dim).
        """
        return replace_nans(actions)

    def get_actions(self):
        return self._actions

    def get_features(self):
        return self._state_features

    def _setup_trainer(self, parameters, state_size, action_size):
        if self.qdrrn_model_type == QDRRN_MODEL_T.DQN \
                or self.qdrrn_model_type == QDRRN_MODEL_T.DQNA:
            self._setup_trainer_dqn(parameters, state_size, action_size)
        elif self.qdrrn_model_type == QDRRN_MODEL_T.DRRN\
                or self.qdrrn_model_type == QDRRN_MODEL_T.KNN_DRRN:
            self._setup_trainer_drrn(parameters, state_size, action_size)
        elif self.qdrrn_model_type == QDRRN_MODEL_T.ACTORCRITIC\
                or self.qdrrn_model_type == QDRRN_MODEL_T.KNN_ACTORCRITIC:
            self._setup_trainer_actorcritic(parameters, state_size, action_size)
        else:
            logger.info("Type not supported " + str(self.qdrrn_model_type))
            self._setup_trainer_dqn(parameters, state_size, action_size)
        if self.qdrrn_model_type == QDRRN_MODEL_T.KNN_DRRN or\
                self.qdrrn_model_type == QDRRN_MODEL_T.KNN_ACTORCRITIC:
            self._setup_trainer_wknn()

    def _setup_trainer_wknn(self):
        if self.qdrrn_model_knn_freq is not None and\
                self.qdrrn_model_knn_k is not None:
            self._trainers[self._trainers_label.index(ACTION_LABEL)]._setup_knn(
                self.qdrrn_model_knn_freq, self.qdrrn_model_knn_k,
                self.qdrrn_model_knn_dynreindex,
                self.qdrrn_model_knn_dynreindex_threshold,
                self.qdrrn_model_knn_dynreindex_rand_other
            )
        else:
            logger.info("WARNING: using default knn setting")
            actiontrainer_ind = self._trainers_label.index(ACTION_LABEL)
            self._trainers[actiontrainer_ind]._setup_knn()

    def _setup_trainer_dqn(self, parameters, state_size, action_size):
        if parameters.training.layers[-1] != 1:
            raise Exception(
                "The last entry of parameters.training.layers should be 1: \
                this trainer is intended to produce a scalar output."
            )
        if parameters.training.layers[0] is None or\
           parameters.training.layers[0] == -1:
            parameters.training.layers[0] = state_size + action_size

        self._trainers_label = [Q_LABEL]
        self._trainers = [
            ContinuousActionDQNTrainer(trainer_label, parameters, action_size)
            for trainer_label in self._trainers_label
        ]

    def _setup_trainer_drrn(self, parameters, state_size, action_size):

        # network at least 2 layers otherwise cannot join same dimension
        assert (len(parameters.training.layers) > 2)
        q_scale = 1.0
        q_scale_normalize = False or q_scale != 1.0
        q_scale_activation = [str(q_scale)] if q_scale_normalize else []

        s_node = [state_size] + parameters.training.layers[1:-1]
        s_actv = parameters.training.activations[:-1] +\
            q_scale_activation
        training = copy.deepcopy(parameters.training)
        training.layers = s_node
        training.activations = s_actv
        parameters_state = DiscreteActionModelParameters(training=training)

        a_node = [action_size] + parameters.training.layers[1:-1]
        a_actv = s_actv
        training = copy.deepcopy(parameters.training)
        training.layers = a_node
        training.activations = a_actv
        parameters_action = DiscreteActionModelParameters(training=training)

        self._trainers_label = [STATE_LABEL, ACTION_LABEL]
        self._trainers = [
            TwoTowerStateTrainer(
                self._trainers_label[0], parameters_state, action_size
            ),
            TwoTowerActionTrainer(
                self._trainers_label[1], parameters_action, action_size
            )
        ]

    def _setup_trainer_actorcritic(self, parameters, state_size, action_size):
        # state: the actor network : state -> action_emb
        # action:  the action embedding network, might keep it freeze for while
        #    or forever, it is back to a regular actor-critic if action network
        #    is just identity mapping meaning action x I -> action_emb
        # critic: is the critic network: (state , action_emb) -> Q value

        # Tricky: part is how to define the "reward or loss on state and action"

        if self.qdrrn_model_type == QDRRN_MODEL_T.KNN_ACTORCRITIC:
            act_emb_size = parameters.training.layers[-2]
        else:
            act_emb_size = action_size
        if parameters.training.layers[0] is None or\
           parameters.training.layers[0] == -1:
            parameters.training.layers[0] = state_size + act_emb_size
        print("NN structure", parameters.training.layers)
        c_node = [state_size + act_emb_size] + \
            parameters.training.layers[1:]
        c_actv = parameters.training.activations
        training = copy.deepcopy(parameters.training)
        training.layers = c_node
        training.activations = c_actv
        parameters_critic = DiscreteActionModelParameters(training=training)

        s_node = [state_size] + parameters.training.layers[1:-2] +\
            [act_emb_size]
        s_actv = parameters.training.activations[:-1]
        training = copy.deepcopy(parameters.training)
        training.layers = s_node
        training.activations = s_actv
        parameters_state = DiscreteActionModelParameters(training=training)

        if self.qdrrn_model_type == QDRRN_MODEL_T.KNN_ACTORCRITIC:
            a_node = [action_size] + \
                parameters.training.layers[1:-1]
            a_actv = parameters.training.activations[:-1]
        else:
            # force action network to be identity? or a fixed parameterized embedding?
            a_node = [action_size]
            a_actv = []
        training = copy.deepcopy(parameters.training)
        training.layers = a_node
        training.activations = a_actv
        parameters_action = DiscreteActionModelParameters(training=training)

        self._trainers_label = [STATE_LABEL, ACTION_LABEL, CRITIC_LABEL]
        critic_trainer = ContinuousActionDQNTrainer(
            self._trainers_label[2],
            parameters_critic,
            action_size,
            as_Critic=True
        )
        self._trainers = [
            TwoTowerStateTrainer(
                self._trainers_label[0], parameters_state, action_size,
                critic_trainer
            ),
            TwoTowerActionTrainer(
                self._trainers_label[1], parameters_action, action_size,
                critic_trainer
            ), critic_trainer
        ]

    def stream_tdp(self, df, evaluator=None):
        """Load large batch as training set. This batch will further be broken
        down into minibatches

        :param df a batch of data.
        """
        is_terminal = np.array(
            [next_action.max() == 0.0 for next_action in df.next_action],
            dtype=np.bool
        )
        possible_next_actions = df.possible_next_actions
        return self.stream(
            self._prepare_states(df.state_features),
            self._prepare_actions(df.action), df.reward,
            self._prepare_states(df.next_state_features),
            self._prepare_actions(df.next_action), is_terminal,
            possible_next_actions, evaluator
        )

    def stream(
        self,
        states,
        actions,
        rewards,
        next_states,
        next_actions,
        is_terminal,
        possible_next_actions,
        evaluator=None
    ):
        """Load large batch as training set. This batch will further be broken
        down into minibatches
        """
        actions = np.array(actions, dtype=np.float32)
        current_states = np.array(states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        next_actions = np.array(next_actions, dtype=np.float32)
        is_terminal = np.array(is_terminal, dtype=np.bool)
        possible_next_actions = np.array(possible_next_actions)

        # note: no need for buffer because no imbalance for different trainer
        all_indices = np.arange(rewards.shape[0])
        size_ext = self.minibatch_size - rewards.shape[0] % self.minibatch_size
        all_indices = np.append(
            np.arange(rewards.shape[0]),
            np.random.randint(rewards.shape[0], size=size_ext)
        )
        np.random.shuffle(all_indices)
        for batch_start in six.moves.range(
            0, rewards.shape[0], self.minibatch_size
        ):
            batch_end = batch_start + self.minibatch_size
            # min(batch_start + self.minibatch_size, len(all_indices))
            indices = all_indices[batch_start:batch_end]

            for label, trainer in zip(self._trainers_label, self._trainers):
                evaluation = trainer.train_batch(
                    self._trainers,
                    current_states[indices],
                    actions[indices],
                    rewards[indices],
                    next_states[indices],
                    next_actions[indices],
                    is_terminal[indices],
                    possible_next_actions[indices],
                    iteration=self._iteration
                )
                if evaluator is not None:
                    if (label == STATE_LABEL and len(self._trainers_label) == 2)\
                            or label == FINAL_OUTPUT_LABEL:
                        evaluator.report(indices, evaluation)
            self._iteration += 1

    def predictor(self):
        """Builds a ContinuousActionPredictor using the networks trained by this Trainer
        """
        return ContinuousActionPredictor.from_trainers(
            self._trainers, self._state_features, self._actions,
            self._state_normalization_parameters,
            self._action_normalization_parameters, self._trainers_label
        )
