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

<img src="https://lookaside.internalfb.com/intern/diff/file/data/?number=760388037" width="50%" height="50%">



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
- `mlp_out_with_ones` is the output from the deep_represent module (with extra column of ones appended for the bias term), also it is the input to the LinUCB module,
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
        self.update_params(self.scorer.mlp_out_with_ones.detach(), batch.reward, batch.weight)
        return loss
```


### LinUCB

#### Background on UCB

UCB (Upper Confidence Bound) is a general algorithm for solving the multi-armed bandit problem.
The algorithm of UCB is depicted as below, where  the first item is the predicted reward of pulling this arm, and the second item is the uncertainty.
 :

<img src="https://lookaside.internalfb.com/intern/diff/file/data/?number=927382569" width="40%" height="40%">



With more “pulls” on a bandit arm, we get more information because we can observe more rewards and as a result we are more confident about our prediction.
For example, if we tried 100 times on some arm and got 80 wins, then the Maximum Likelihood Estimation (MLE) is 0.80, which is large. As a result we want to exploit it, because we know the chance of winning is as good as 80%. This corresponds to the first item, Qt(a) in the formula.


On the other hand, on another arm we only tried 2 times and got 1 win, then the MLE (first item of UCB) is as small as 0.5. Though 50% chance of win is not good, to be fair we only tried twice after all. This arm may actually be very good and we just do not know yet because we did not try enough times. Thus, it is worth taking some risk and trying more on this arm. This is exploration, and corresponds to denominator Nt(a).
This idea is shown in the below toy example :


<img src="https://lookaside.internalfb.com/intern/diff/file/data/?number=927429214" width="40%" height="40%">



#### LinUCB Model

LinUCB is a more complex version of UCB and can be applied to contextual bandit problems.
Usually it is shortly called as LinUCB because it is a Linear model that outputs Upper Confidence Bound.
In LinUCB, the expected reward of an arm is linear in its context covariates at time t, i.e.,

`reward = w * feature + bias`

where `w` is the linear combination parameters. LinUCB is mostly known as first published in https://arxiv.org/pdf/1003.0146.pdf ,
but the idea behind it can be tracked even further back to linear regression with Bayesian confidence (uncertainty).
The concept of confidence (uncertainty) is depicted in the algorithm below.
The first item is predicted reward which corresponds to the first item of UCB.
The second item is the uncertianty which correponds to the second item of UCB.
The difference compared against UCB is that both predicted reward and uncertainty are calculated with the assistance of feature.
It is feature that allows different arms being able to share their mutual information. In practice, e.g., this means cold start problem can be relieved by adopting LinUCB.
Note that in a classical supervised Linear Regression model without uncertainty, "arms" also can share mutual information but only in the predicted reward they share information.
LinUCB allows arms to share information also on the uncertainty, which further helps to relieve cold start problem.




<img src="https://lookaside.internalfb.com/intern/diff/file/data/?number=927315010" width="40%" height="40%">


In our implementation,
- `A` is basically the sum of the outer product of arms features, i.e., `torch.matmul(x.t(), x)`. Its  complete version of code is as below:

```
self.scorer.cur_avg_A = (
            self.scorer.cur_avg_A * (1 - batch_sum_weight / self.scorer.cur_sum_weight)
            + torch.matmul(x.t(), x * weight) / self.scorer.cur_sum_weight
        )  # dim (DA*DC, DA*DC)
```

- `b` contains the label(i.e., reward) information: `torch.matmul(x.t(), y)` where `y` is the reward.

        self.scorer.cur_avg_b = (
            self.scorer.cur_avg_b * (1 - batch_sum_weight / self.scorer.cur_sum_weight)
            + torch.matmul(x.t(), y * weight).squeeze() / self.scorer.cur_sum_weight
        )  # dim (DA*DC,)


- Instead of being trained by a PyTorch optimizer with gradient back propogation,
we analytically update attributes `A, b`, and then
calculate the Linear Regression model weights by using `A, b` :
```
self._coefs = torch.matmul(self.inv_avg_A, self.avg_b)
```

This way the uncertainty can be obtained from `A` by the quardratic form
```
torch.matmul(x, A) * x
```


#### LinUCB Trainer
Below is a pseudo code of the LinUCB Trainer :

```
class LinUCBTrainer(...):
    def update_params(...):
        self.scorer.cur_avg_A = (... torch.matmul(x.t(), x) ... # update A
        self.scorer.cur_avg_b = (... torch.matmul(x.t(), y) ... # update b

    def cb_training_step(...):
        x = _get_chosen_arm_features(batch.context_arm_features, batch.action)
        # each training step, parameters are updated by using feature & action, where action is the chosen arm.
        # Usually the chosen arm is the arm with largest UCB score.
        self.update_params(x, batch.reward, batch.weight)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # at the end of the training epoch calculate the coefficients
        self.scorer._calculate_coefs()
```

This trainer controls the training process, and updates the model parameters.
