# Multi-Arm and Contextual Bandits

## Overview
Multi-armed bandit (MAB)  is a type of reinforcement learning problem  that deals with decision-making. It is for solving problems where an agent has to make decisions based on uncertain outcomes.
An agent has to choose between multiple options (arms) and receive a reward for each choice. The goal is to maximize the total reward over time by choosing the best arm.

Reinforcement learning (RL) is a broader class of problems that includes MAB as a special case. In RL, an agent interacts with an environment and receives feedback in the form of rewards. The goal is to learn a policy that maximizes the expected cumulative reward over time. MAB can be considered as a simpler version of RL where there is only one state and no transition between states. RL can be used to more complex scenarios where there are multiple states and transitions.

Further, contextual bandit are a type of multi-armed bandit problem where the rewards depend on the context. In Contextual MAB, an agent has to choose between multiple options (arms) based on the context and receive a reward for each choice. The goal is still to maximize the total reward over time by choosing the best arm for each context. Contextual MAB has many applications in various fields such as  online advertising and recommendation systems. Contextual MAB has found solid applications in our team's projects such as Ads Container Selection  where we have context, user features and ad features available for usage. These features can boost model performance comparing against non-contextual MAB. Based on this background, we had implemented some popular and promising Contextual MAB algorithms.

- Ads Container Selection project : https://fburl.com/gslide/9ohxr4mc
- Instagram Reels Exploration project : https://fb.workplace.com/notes/622173379301233

## Contextual MAB Algorithms Supported


### Neural Network LinUCB (NNLinUCB)
#### Model

![alt text](https://lookaside.internalfb.com/intern/diff/file/data/?number=760388037)

This NNLinUCB model (`DeepRepresentLinRegressionUCB`) is a multi-layer regression model that outputs UCB score.
There are two modules in this model: MLP module and LinUCB module.
- The MLP module consists of bottom layers whic are trainable by torch optimizer().
- The LinUCB module is the last layer and it is not updated by optimizer but by matrix computations. The reason to use matrix computations to update model parameters is to output uncertainty besides prediction.
- The Deep Represent (i.e., MLP) module refers to paper https://arxiv.org/pdf/2012.01780.pdf.
- The LinUCB module refers to paper https://arxiv.org/pdf/2012.01780.pdf.


Model Example :

    Features(dim=9) --> deep_represent_layers --> Features(dim=3) --> LinUCB --> ucb score

    DeepRepresentLinUCBTrainer(
    (scorer): DeepRepresentLinearRegressionUCB(
        (deep_represent_layers): FullyConnectedNetwork(
        (dnn): Sequential(
            (0): Linear(in_features=9, out_features=6, bias=True)
            (1): ReLU()
            (2): Linear(in_features=6, out_features=3, bias=True)
            (3): Identity()
        )
        )
    )
    (loss_fn): MSELoss()
    )

In this implementation,
- `pred_u` is the predicted reward,
- `pred_sigma` is the uncertainty associated with the predicted reward,
- `mlp_out` is the output from the deep_represent module, also it is the input to the LinUCB module,
- `coefs` serve as the top layer LinUCB module in this implementation
    - it is crucial that `coefs` will not be updated by gradient back propagation
    - coefs is defined by `@property` decorator in `LinearRegressionUCB`
- `ucb_alpha` controls the balance of exploration and exploitation,
    and it also indicates whether `pred_sigma` will be included in the final output:
    - If `ucb_alpha` is not 0, `pred_sigma` will be included in the final output.
    - If `ucb_alpha` is 0, `pred_sigma` will not be included and it is equivalent to a classical supervised MLP model.

#### Trainer
The `DeepRepresentLinUCBTrainer` is the trainer that is associated with `DeepRepresentLinRegressionUCB` model, and it can be considered as an extension to the `LinUCBTrainer`.
Below is a mini pseudocode of the implementation of the trainer :
```
class DeepRepresentLinUCBTrainer(LinUCBTrainer):
        self.scorer = policy.scorer
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.optimizer = optimizer

    def training_step(self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0):
        pred_ucb = self.scorer(inp=x)
        loss = self.loss_fn(self.scorer.pred_u, batch.reward.t())
        self.update_params(self.scorer.mlp_out.detach(), batch.reward, batch.weight)
        return loss
```


### LinUCB
- https://arxiv.org/pdf/1003.0146.pdf
