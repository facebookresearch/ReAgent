import logging
from itertools import permutations

import reagent.types as rlt
import torch
from reagent.model_utils.seq2slate_utils import Seq2SlateOutputArch
from reagent.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import Seq2SlateParameters
from reagent.torch_utils import gather
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer


logger = logging.getLogger(__name__)


MODEL_TRANSFORMER = "transformer"
ON_POLICY = "on_policy"
SIMULATION = "simulation"


def create_trainer(seq2slate_net, learning_method, batch_size, learning_rate, device):
    use_gpu = False if device == torch.device("cpu") else True
    if learning_method == ON_POLICY:
        return Seq2SlateTrainer(
            seq2slate_net=seq2slate_net,
            minibatch_size=batch_size,
            parameters=Seq2SlateParameters(on_policy=True),
            policy_optimizer=Optimizer__Union.default(lr=learning_rate),
            use_gpu=use_gpu,
            print_interval=100,
        )


def create_seq2slate_net(
    model_str,
    candidate_num,
    candidate_dim,
    hidden_size,
    output_arch,
    temperature,
    device,
):
    if model_str == MODEL_TRANSFORMER:
        return Seq2SlateTransformerNet(
            state_dim=1,
            candidate_dim=candidate_dim,
            num_stacked_layers=2,
            num_heads=2,
            dim_model=hidden_size,
            dim_feedforward=hidden_size,
            max_src_seq_len=candidate_num,
            max_tgt_seq_len=candidate_num,
            output_arch=output_arch,
            temperature=temperature,
        ).to(device)
    else:
        raise NotImplementedError(f"unknown model type {model_str}")


def post_preprocess_batch(
    learning_method, seq2slate_net, candidate_num, batch, device, epoch
):
    if learning_method == ON_POLICY:
        model_propensity, model_action, reward = rank_on_policy_and_eval(
            seq2slate_net, batch, candidate_num, greedy=False
        )
        on_policy_batch = rlt.PreprocessedRankingInput.from_input(
            state=batch.state.float_features,
            candidates=batch.src_seq.float_features,
            device=device,
            action=model_action,
            logged_propensities=model_propensity,
            slate_reward=-reward,  # negate because we want to minimize
        )
        logger.info(f"Epoch {epoch} mean on_policy reward: {torch.mean(reward)}")
        logger.info(
            f"Epoch {epoch} mean model_propensity: {torch.mean(model_propensity)}"
        )
        return on_policy_batch
    return batch


def create_batch(batch_size, candidate_num, candidate_dim, device, diverse_input=False):
    state = torch.zeros(batch_size, 1)  # fake state, we only use candidates
    # # city coordinates are spread in [0, 4]
    candidates = torch.randint(5, (batch_size, candidate_num, candidate_dim)).float()
    if not diverse_input:
        # every training data has the same nodes as the input cities
        candidates[1:] = candidates[0]
    batch = rlt.PreprocessedRankingInput.from_input(
        state=state.to(device), candidates=candidates.to(device), device=device
    )
    return batch


def create_train_and_test_batches(
    batch_size, candidate_num, candidate_dim, device, num_train_batches, diverse_input
):
    train_batches = [
        create_batch(
            batch_size,
            candidate_num,
            candidate_dim,
            device,
            diverse_input=diverse_input,
        )
        for _ in range(num_train_batches)
    ]

    if diverse_input:
        test_batch = create_batch(
            batch_size,
            candidate_num,
            candidate_dim,
            device,
            diverse_input=diverse_input,
        )
    else:
        test_batch = train_batches[0]

    return train_batches, test_batch


def compute_reward(ranked_cities):
    assert len(ranked_cities.shape) == 3
    ranked_cities_offset = torch.roll(ranked_cities, shifts=1, dims=1)
    return (
        torch.sqrt(((ranked_cities_offset - ranked_cities) ** 2).sum(-1))
        .sum(-1)
        .unsqueeze(1)
    )


def compute_best_reward(input_cities):
    batch_size, candidate_num, _ = input_cities.shape
    all_perm = torch.tensor(
        list(permutations(torch.arange(candidate_num), candidate_num))
    )
    res = [
        compute_reward(gather(input_cities, perm.repeat(batch_size, 1)))
        for perm in all_perm
    ]
    # res shape: batch_size, num_perm
    res = torch.cat(res, dim=1)
    best_possible_reward = torch.min(res, dim=1).values
    best_possible_reward_mean = torch.mean(best_possible_reward)
    return best_possible_reward_mean


@torch.no_grad()
def rank_on_policy(
    model, batch: rlt.PreprocessedRankingInput, tgt_seq_len: int, greedy: bool
):
    model.eval()
    rank_output = model(
        batch, mode=Seq2SlateMode.RANK_MODE, tgt_seq_len=tgt_seq_len, greedy=greedy
    )
    ranked_slate_prob = rank_output.ranked_per_seq_probs
    ranked_order = rank_output.ranked_tgt_out_idx - 2
    model.train()
    return ranked_slate_prob, ranked_order


@torch.no_grad()
def rank_on_policy_and_eval(
    seq2slate_net, batch: rlt.PreprocessedRankingInput, tgt_seq_len: int, greedy: bool
):
    model_propensity, model_action = rank_on_policy(
        seq2slate_net, batch, tgt_seq_len, greedy=greedy
    )
    ranked_cities = gather(batch.src_seq.float_features, model_action)
    reward = compute_reward(ranked_cities)
    return model_propensity, model_action, reward


def run_seq2slate_tsp(
    model_str,
    batch_size,
    epochs,
    candidate_num,
    num_batches,
    hidden_size,
    diverse_input,
    learning_rate,
    expect_reward_threshold,
    learning_method,
    device,
):
    candidate_dim = 2
    eval_sample_size = 1

    train_batches, test_batch = create_train_and_test_batches(
        batch_size, candidate_num, candidate_dim, device, num_batches, diverse_input
    )
    best_test_possible_reward = compute_best_reward(test_batch.src_seq.float_features)

    seq2slate_net = create_seq2slate_net(
        model_str,
        candidate_num,
        candidate_dim,
        hidden_size,
        Seq2SlateOutputArch.AUTOREGRESSIVE,
        1.0,
        device,
    )

    trainer = create_trainer(
        seq2slate_net, learning_method, batch_size, learning_rate, device
    )

    for e in range(epochs):
        # training
        for batch in train_batches:
            batch = post_preprocess_batch(
                learning_method, seq2slate_net, candidate_num, batch, device, e
            )
            trainer.train(rlt.PreprocessedTrainingBatch(training_input=batch))

        # evaluation
        best_test_reward = torch.full((batch_size,), 1e9).to(device)
        for _ in range(eval_sample_size):
            _, _, reward = rank_on_policy_and_eval(
                seq2slate_net, test_batch, candidate_num, greedy=True
            )
            best_test_reward = torch.where(
                reward < best_test_reward, reward, best_test_reward
            )
        logger.info(
            f"Test mean reward: {torch.mean(best_test_reward)}, "
            f"best possible reward {best_test_possible_reward}"
        )
        if (
            torch.mean(best_test_reward)
            < best_test_possible_reward * expect_reward_threshold
        ):
            return

    raise AssertionError("Test failed because it did not reach expected test reward")
