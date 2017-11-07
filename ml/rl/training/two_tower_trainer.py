from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import six
import time

# @build:deps [
# @/caffe2/caffe2/fb:log_file_db
# @/caffe2/caffe2/python:caffe2_py
# @/deeplearning/projects/faiss:pyfaiss
# ]
# if need to include faiss_gpu:
# # @/deeplearning/projects/faiss:pyfaiss_gpu

import logging
logger = logging.getLogger(__name__)

from ml.rl.training.ml_trainer_extension import \
    MLTrainerIP, MLTrainerExt
from ml.rl.training.target_network import \
    TargetNetwork

from ml.rl.training.knn_engine import KnnEngine,\
    fetch_new_rows_toappend

Q_LABEL = "Q"
CRITIC_LABEL = "Q"
STATE_LABEL = "State"
ACTION_LABEL = "Action"
FINAL_OUTPUT_LABEL = CRITIC_LABEL

DEFAULT_USE_PRESERVE = True


def pprint_perct(anum):
    return "{0:.2f}%".format(100 * anum)


def pprint_perct_list(alist):
    # pretty proint list of percentages
    return ', '.join([pprint_perct(a) for a in alist])


class TwoTowerStateTrainer(object):
    """ This class implements TD training to train a network for state only
    as input, and certain action or embedding dim as output
    """

    def __init__(
        self,
        name,
        problem_definition,
        rl_parameters,
        num_action_features,
        extension_critic=None
    ):
        self.num_action_features = num_action_features
        self.extension_critic = extension_critic
        self.nn_innerproduct_or_l2dist = "IP"  # meaning ip, otherwise l2
        if extension_critic is not None:
            self.nn_innerproduct_or_l2dist = "L2"

        self._general_setup(problem_definition, rl_parameters)
        self._overall_trainer_setup(name, rl_parameters)
        self._reset_preserves()

    def _reset_preserves(self):
        self._preserve_labels = None
        self._preserve_state_emb = None

    def _overall_trainer_setup(self, name, rl_parameters):
        if self.nn_innerproduct_or_l2dist == "IP":
            self._trainer = MLTrainerIP(
                name + self._trainer_label,
                rl_parameters.network.nodes,
                rl_parameters.network.activations,
                rl_parameters.training.minibatch_size,
                rl_parameters.training.learning_rate,
                rl_parameters.training.optimizer,
                scaled_output=True
            )
        else:
            self._trainer = MLTrainerExt(
                name + self._trainer_label,
                rl_parameters.network.nodes,
                rl_parameters.network.activations,
                rl_parameters.training.minibatch_size,
                rl_parameters.training.learning_rate,
                rl_parameters.training.optimizer,
                scaled_output=False,
                extension_mltrainer=self.extension_critic._trainer
            )
        self._target_network = TargetNetwork(
            self._trainer, rl_parameters.training.target_update_rate
        )

    def _labels(
        self, all_trainers, rewards, next_states, next_actions, is_terminal,
        possible_next_actions, iteration
    ):
        labels = np.copy(rewards[:, np.newaxis])

        next_state_embedding = self._target_network.target_values(next_states)

        found_action_trainer = self._get_action_trainer(all_trainers)
        next_action_embedding = found_action_trainer._target_network.\
            target_values(next_actions)
        external_input = next_action_embedding\
            if self.nn_innerproduct_or_l2dist == 'IP' else next_states
        if self._maxq_learning is False:
            q_val_next_s = self._trainer.score_wexternal(
                next_state_embedding, external_input
            )

            q_update = np.where(is_terminal, 0, q_val_next_s)
            labels[:, 0] += q_update * self._gamma
        else:
            # hand the select best action to action trainer ...
            labels = found_action_trainer._labels(
                all_trainers, rewards, next_states, next_actions, is_terminal,
                possible_next_actions, iteration
            )
        return labels

    def train_batch(
        self, all_trainers, states, actions, rewards, next_states, next_actions,
        is_terminal, possible_next_actions, iteration
    ):
        if iteration == self._reward_burnin:
            self._target_network.enable_slow_updates()
        labels = self._labels(
            all_trainers, rewards, next_states, next_actions, is_terminal,
            possible_next_actions, iteration
        )

        found_action_trainer = self._get_action_trainer(all_trainers)
        action_embedding = found_action_trainer._target_network.target_values(
            actions
        )
        external_input = action_embedding\
            if self.nn_innerproduct_or_l2dist == 'IP' else states
        evaluation_state = self._trainer.train_batch_wexternal(
            states, labels, external_input, evaluate=True
        )

        self._preserve_labels = labels
        self._preserve_state_emb = self._trainer.output

        self._target_network.target_update()
        return evaluation_state

    def build_predictor_ext(self, *args, **kwargs):
        return self._trainer.build_predictor_ext(*args, **kwargs)

    def extend_predictor(self, *args, **kwargs):
        self._trainer.extend_predictor_only(*args, **kwargs)

    @property
    def loss(self):
        return self._trainer.loss


def act_hash(act_vector):
    # temporarily using the array itself
    return str(act_vector)


# Note: this is a simple replacement for np.unique(a, axis=0) since
#  axis is supported after numpy 1.3.7 where ours is 1.3.1
def unique_rows(data):
    sorted_data = data[np.lexsort(data.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
    return sorted_data[row_mask]


class TwoTowerActionTrainer(TwoTowerStateTrainer):
    """ This class implements TD training to train a network for action only
    as input, and certain action embedding dim as output
    """

    def __init__(
        self,
        name,
        problem_definition,
        rl_parameters,
        num_action_features,
        extension_critic=None
    ):
        self.num_action_features = num_action_features
        self.extension_critic = extension_critic
        self.nn_innerproduct_or_l2dist = "IP"  # meaning ip, otherwise l2
        self._as_actor = False
        if extension_critic is not None:
            self.nn_innerproduct_or_l2dist = "L2"
            self._as_actor = True
        self._general_setup(problem_definition, rl_parameters)
        self._overall_trainer_setup(name, rl_parameters)

        self._act_embedding_dim = rl_parameters.network.nodes[-1]
        self._act_dim = rl_parameters.network.nodes[0]

        self.max_possible_count = 0
        self.max_possible_count_threshold = 100
        self.max_possible_count_unchanged_times = 0
        self.max_possible_count_unchanged_times_tolerance = 1000

        # knn stuff prepare
        self._knn_refresh_iter_freq = -1
        self._knn_search_k = -1
        self._knn_faiss_engine = None

        self._cached_action_raw = np.array([]).reshape((0, self._act_dim))
        self._cached_action_raw = self._cached_action_raw.astype(np.float32)
        self._cached_action_emb = np.array([]).reshape(
            (0, self._act_embedding_dim)
        ).astype(np.float32)
        self._cached_action_raw_MAXLENGTH = 1000000

        self._cached_action_insertfail_times = 0
        self._cached_action_insertfail_times_tolorance = 10
        self._knn_eval_action_minibatch_size = 10240

        self._knn_refresh_times = -1
        self._non_knntopk_pert_record = []
        # keep track of percentage of times top1 is not best option, tracing down knn
        # can be used as threshold for re-start knn indexing since prev tree not well

    def set_all_possible_actions(self, all_possible_actions):
        if self.all_possible_actions is not None or all_possible_actions is None:
            return
        self.all_possible_actions = all_possible_actions
        self.max_possible_count = len(all_possible_actions)
        self.max_possible_count_unchanged_times_tolerance = -1
        self._cached_action_insertfail_times_tolorance = -1
        self._cached_action_raw = self.all_possible_actions

    def _setup_knn(
        self,
        knn_refresh_iter_freq=10,
        knn_search_k=1,
        non_knntopk_pert_trigger_reindex=False,
        non_knntopk_pert_threshold=0.5,
        non_knntopk_random_choice=3
    ):
        if knn_refresh_iter_freq > 0 and knn_search_k > 0:
            if not self._maxq_learning:
                logger.info("KNNWarning: CHECK YOUR SETTING: knn but not max_Q")

        self._nontopk_checking_freq = 2
        self._nontopk_warning_window = 3
        self._nontopk_warning_threshold = non_knntopk_pert_threshold
        self._non_knntopk_pert_record = []
        self._non_knntopk_pert_exceed = True
        self._non_knntopk_pert_trigger_reindex = non_knntopk_pert_trigger_reindex
        self._non_knn_randm = non_knntopk_random_choice

        self._knn_refresh_iter_freq = knn_refresh_iter_freq
        self._knn_search_k = knn_search_k
        logger.info(
            "Setup KNN tree, and init index " + self.nn_innerproduct_or_l2dist +
            ", knn_refresh_freq = " + str(self._knn_refresh_iter_freq) +
            ", knn_k = " + str(self._knn_search_k)
        )
        if self._non_knntopk_pert_trigger_reindex:
            logger.info(
                "Setup KNN dynamic refresh: sampling {} from non-topk, ".
                format(self._non_knn_randm) +
                "threshold on percentage of nontopk better than best of topk" +
                " > " + pprint_perct(self._nontopk_warning_threshold)
            )
        assert self._knn_search_k > 0
        if self._knn_refresh_iter_freq > 0:
            self._knn_faiss_engine = KnnEngine(
                self._act_embedding_dim, self._cached_action_raw_MAXLENGTH,
                self._knn_search_k, self.nn_innerproduct_or_l2dist
            )
        self._preserved_knn_actions_raw = None

    # Dedicated to KNN begin
    def _refresh_eval_act_index_knn(self, iteration):
        # time to build the index?
        if self._knn_refresh_times >= 0 and \
            self._cached_action_insertfail_times < \
                self._cached_action_insertfail_times_tolorance:
            return
        # note: check if current iteraiton already build knn to avoid multiple
        #  update per round
        if self._knn_refresh_times >= 0 and self._knn_refresh_times == iteration:
            return

        # if we use topk-vs-nontopk comparision as standard to refresh or not
        if not self._non_knntopk_pert_trigger_reindex:
            # if all pass, just see if curr iter need knn refresh
            if iteration % self._knn_refresh_iter_freq != 0:
                return
        else:
            if not self._non_knntopk_pert_exceed:
                return

        logger.info(
            "KNNWarning: eval action and build index for {} data".
            format(self._cached_action_raw.shape[0]) + ", might take long time"
        )
        t = time.time()
        # re-generate the overall action_embbedding based on acion cache
        self._cached_action_emb = np.array([]).reshape(
            (0, self._act_embedding_dim)
        )
        for batch_start in six.moves.range(
            0, self._cached_action_raw.shape[0],
            self._knn_eval_action_minibatch_size
        ):
            batch_end = min(
                [
                    batch_start + self._knn_eval_action_minibatch_size,
                    self._cached_action_raw.shape[0]
                ]
            )
            batch_action_emb = self._target_network.target_values(
                self._cached_action_raw[batch_start:batch_end]
                .astype(np.float32)
            )

            self._cached_action_emb = np.concatenate(
                [self._cached_action_emb, batch_action_emb]
            )
        # now send it to knn index
        success = self._knn_faiss_engine.build_index_knn(
            self._cached_action_emb
        )
        t = time.time() - t
        logger.info(
            "KNNWarning: eval action and build index for {} data finished, ".
            format(self._cached_action_raw.shape[0]) +
            ", taking {0:.5f} s.".format(t)
        )

        if success:
            self._knn_refresh_times = iteration
            if self._knn_search_k > 1:
                if len(self._non_knntopk_pert_record) > 0:
                    logger.info(
                        "KNN_INFO: previous percentage of picking non-topk: " +
                        pprint_perct_list(self._non_knntopk_pert_record[-10:])
                    )
                    self._non_knntopk_pert_record = []  # restart tracking
                    self._non_knntopk_pert_exceed = False
            logger.info("KNN_INFO: knn index built")
        else:
            logger.info("KNNWarning: knn build index failed")
        return success

    def _find_knn_best_distances(self, embbeding_queries):
        if self._knn_faiss_engine.check_engine_build():
            dist, _ = self._knn_faiss_engine.find_knn_best_dist_ind(
                embbeding_queries
            )
            return dist
        else:
            return None

    def _find_knn_best_instances(self, embbeding_queries):
        if self._knn_faiss_engine.check_engine_build():
            _, indx = self._knn_faiss_engine.find_knn_best_dist_ind(
                embbeding_queries
            )
            return self._cached_action_emb[indx]
        else:
            return None

    def _find_knn_k_rawact_intances(self, embbeding_queries):
        if self._knn_faiss_engine.check_engine_build():
            _, indexs = self._knn_faiss_engine.find_knn_dist_ind(
                embbeding_queries
            )
            # indexes: (batch x k) -> _cached_action_raw (batch x k x action_dim)
            return self._cached_action_raw[indexs].astype(np.float32)
        else:
            return None

    def _find_m_rand_rawact_intances(self, batch, m):
        randindx = np.random.choice(len(self._cached_action_raw),
                                    batch * m).reshape(batch, m)
        return self._cached_action_raw[randindx].astype(np.float32), randindx

    def _update_action_cache(self, possible_next_actions):
        # no need to check if we can update cache again
        if self._cached_action_insertfail_times > \
                self._cached_action_insertfail_times_tolorance or \
                self._cached_action_insertfail_times_tolorance < 0:
            return

        # special case check: only none or all possible action included
        # in which case we dont do the complex insert again
        possible_next_actions_useful = [
            pna
            for pna in possible_next_actions if pna is not None and len(pna) > 0
        ]
        if len(possible_next_actions_useful) == 0:
            if len(self._cached_action_raw) > 0:
                self._cached_action_insertfail_times += 1
            return

        unique_actions = unique_rows(np.vstack(possible_next_actions_useful))
        unique_actions_new = fetch_new_rows_toappend(
            self._cached_action_raw, unique_actions
        )
        # #repeated_count = len(unique_actions) - len(unique_actions_new)
        # self._cached_action_insertfail_times += or += repeated_count
        self._cached_action_insertfail_times += (len(unique_actions_new) == 0)

        # is replacement really needed? when we are overflow with data
        if len(self._cached_action_raw) > self._cached_action_raw_MAXLENGTH:
            self._cached_action_raw = np.random.\
                choice(self._cached_action_raw,
                       self._cached_action_raw_MAXLENGTH - 10, replace=False)

        self._cached_action_raw = np.concatenate(
            [self._cached_action_raw, unique_actions_new]
        )

        # # after tree build, if we actually need to update it by add more data?
        # if len(unique_actions_new) > 0 and\
        #         self._knn_faiss_engine.check_engine_build():
        #     self._knn_faiss_engine.append_index_knn(self._target_network.
        #                                             target_values(unique_actions_new))
        return len(unique_actions_new)

    # Dedicated to KNN end

    def _update_max_possible_count(self, possible_next_actions):
        # this should be persistent in the entire training, so no need to reset
        if self.max_possible_count_unchanged_times_tolerance < 0 or \
            self.max_possible_count_unchanged_times > \
                self.max_possible_count_unchanged_times_tolerance:
            return

        max_possible_list = [
            len(a) for a in possible_next_actions if a is not None
        ]
        max_possible = 0
        if len(max_possible_list) > 0:
            max_possible = max(max_possible_list)

        max_possible = min([max_possible, self.max_possible_count_threshold])
        if self.max_possible_count >= max_possible:
            self.max_possible_count_unchanged_times += 1
        else:
            self.max_possible_count = max_possible

    def _labels(
        self, all_trainers, rewards, next_states, next_actions, is_terminal,
        possible_next_actions, iteration
    ):

        labels = np.copy(rewards[:, np.newaxis])
        found_state_trainer = self._get_state_trainer(all_trainers)
        next_state_embedding = found_state_trainer._target_network.\
            target_values(next_states)
        next_action_embedding = self._target_network.target_values(next_actions)
        external_input = next_state_embedding if not self._as_actor else next_states

        if self._maxq_learning is False:
            # sarsa
            q_val_next_a = self._trainer.score_wexternal(
                next_action_embedding, external_input
            )
        else:  # self._maxq_learning
            # define a genereic search best among all possible/knn
            def search_through_candidate_actions(
                candidate_next_actions, repeated_count
            ):
                repeated_next_state_external_input = repeat_states(
                    next_states
                    if self._as_actor else next_state_embedding, repeated_count
                )
                # first extend, then map to embedding
                extend_poss_actions = extend_possible_actions(
                    candidate_next_actions, self.num_action_features,
                    repeated_count
                )
                extend_poss_actions_emb = self._target_network.\
                    target_values(extend_poss_actions)

                next_values_ext = self._trainer.\
                    score_wexternal(extend_poss_actions_emb,
                                    repeated_next_state_external_input)
                next_values_ext = next_values_ext.reshape(
                    (
                        -1,
                        repeated_count,
                    )
                )
                all_next_value_max = np.max(next_values_ext, axis=1)
                all_next_index_max = np.argmax(next_values_ext, axis=1)
                return all_next_value_max, all_next_index_max, next_values_ext

            # end of defnition for search

            if self._knn_refresh_iter_freq <= 0 or self._knn_search_k <= 0:
                # this is to check how max possible q we need to trace
                # update self.max_possible_count
                self._update_max_possible_count(possible_next_actions)
                possible_next_actions = [
                    pna if pna is not None else self.all_possible_actions
                    for pna in possible_next_actions
                ]
                all_next_value, all_next_value_index, _ = \
                    search_through_candidate_actions(possible_next_actions,
                                                     self.max_possible_count)
                q_val_next_a = all_next_value.flatten()

            if self._knn_search_k >= 1:
                self._update_action_cache(possible_next_actions)
                self._refresh_eval_act_index_knn(iteration)
                # necessary to re-compute q for top k rather than using existing
                topknn_next_actions = self._find_knn_k_rawact_intances(
                    next_state_embedding
                )
                if topknn_next_actions is None:
                    q_val_next_a = None
                else:
                    self._preserved_knn_actions_raw = topknn_next_actions.astype(
                        np.float32
                    )
                # below is when topknn_next_action is not None
                if self._knn_search_k == 1 and topknn_next_actions is not None:
                    # special case for k=1 to save time, avoid expand and sort
                    top1_act = topknn_next_actions
                    top1_action_emb = self._target_network.target_values(
                        top1_act
                    )
                    top1_next_value = self._trainer.score_wexternal(
                        top1_action_emb, external_input
                    )
                    q_val_next_a = top1_next_value.flatten()
                if self._knn_search_k > 1 and topknn_next_actions is not None:
                    all_next_value, all_next_value_index, _ = \
                        search_through_candidate_actions(topknn_next_actions,
                                                         self._knn_search_k)

                    # finished:random sample from non-topk compared with best
                    # should refresh index when more non-topk are better
                    if self._nontopk_checking_freq > 0 and \
                            iteration % self._nontopk_checking_freq == 0 and \
                            self._nontopk_warning_threshold > 0 and \
                            self._non_knn_randm > 0:
                        m_randsample_action, m_randsample_index = self.\
                            _find_m_rand_rawact_intances(all_next_value.shape[0],
                                                         self._non_knn_randm)
                        _, _, m_randsample_all_next_val = \
                            search_through_candidate_actions(m_randsample_action,
                                                             self._non_knn_randm)
                        non_knntopk_pert = np.sum(
                            m_randsample_all_next_val -
                            np.expand_dims(all_next_value, 1) > 0
                        ) / float(m_randsample_all_next_val.size)
                        self._non_knntopk_pert_record.append(non_knntopk_pert)
                        # if tracking percentage of times top1 is not best in knn
                        # if self._nontopk_warning_threshold > 0:
                        #     non_knntop1_pert = np.sum(all_next_value_index != 0) /\
                        #         float(all_next_value_index.size)
                        nontopk_sumlimit = self._nontopk_warning_threshold *\
                            self._nontopk_warning_window
                        nontopk_warning_check = self._non_knntopk_pert_record[
                            -self._nontopk_warning_window:
                        ]
                        if sum(nontopk_warning_check) > nontopk_sumlimit:
                            logger.info(
                                "KNNWarning: index might be deprecated since " +
                                "last {} batches with over {} of non-topk: ".
                                format(
                                    str(self._nontopk_warning_window),
                                    pprint_perct(
                                        self._nontopk_warning_threshold
                                    )
                                ) + pprint_perct_list(nontopk_warning_check) +
                                ". Note: not forcing rebuild knn index"
                            )
                            self._non_knntopk_pert_exceed = True

                    q_val_next_a = all_next_value.flatten()

                if q_val_next_a is None:
                    logger.info(
                        "KNNWarning: Falling back to sarsa value: iteration:" +
                        str(iteration)
                    )
                    q_val_next_a = self._trainer.score_wexternal(
                        next_action_embedding, external_input
                    )

        q_update = np.where(is_terminal, 0, q_val_next_a)
        labels[:, 0] += q_update * self._gamma
        return labels

    def train_batch(
        self, all_trainers, states, actions, rewards, next_states, next_actions,
        is_terminal, possible_next_actions, iteration
    ):
        use_preserve = DEFAULT_USE_PRESERVE
        if iteration == self._reward_burnin:
            self._target_network.enable_slow_updates()

        found_state_trainer = self._get_state_trainer(all_trainers)
        preserved_labels = found_state_trainer._preserve_labels
        if use_preserve and preserved_labels is not None:
            labels = preserved_labels
        else:
            labels = self._labels(
                all_trainers, rewards, next_states, next_actions, is_terminal,
                possible_next_actions, iteration
            )

        preserved_state_embedding = found_state_trainer._preserve_state_emb
        if use_preserve and preserved_state_embedding is not None:
            states_embedding = preserved_state_embedding
        else:
            states_embedding = found_state_trainer._target_network.target_values(
                states
            )

        external_input = states_embedding if not self._as_actor else states
        evaluation_action = self._trainer.train_batch_wexternal(
            actions, labels, external_input, evaluate=True
        )

        self._target_network.target_update()

        found_state_trainer._reset_preserves()
        return evaluation_action
