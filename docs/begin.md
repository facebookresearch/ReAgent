---
id: begin
title: BlueWhale: Applied Reinforcement Learning
hide_title: true
sidebar_label: Introduction
---

<br/>
![BlueWhale Logo](../img/logo/logo_no_background.png)
<br/>

Reinforcement Learning (RL) is an area of machine learning concerned with optimizing a cumulative reward in an environment. Some examples of problems where RL is used include robot control, board games, and supply chain management.  At Facebook we use RL in Growth, Marketing, Network Optimization, and Story Ranking services.  In general, we use RL to optimize our recommender systems and our infrastructure.

BlueWhale is a set of production-ready RL workflows and Deep RL algorithms that are fast, well tested, and becnhmarked. These algorithms are built on [PyTorch](http://pytorch.org/) and [caffe2](http://caffe2.ai/).  The platform currently contains:

1. Discrete and Parametric-Action SARSA models.
2. Discrete and Parametric-Action Deep Q Learning (DQN) models.
3. Deep Deterministic Policy Gradients (DDPG).

Internally, we train these models on large databases of episodes, but externally we provide support for running BlueWhale inside [OpenAI Gym](gym.openai.com).  BlueWhale is extremely fast and can train models with thousands of parameters on billions of rows of data.  All of our models export to the caffe2 predictor after training so the trained model can be efficiently served on hundreds of thousands of machines to billions of people.

## RL + Recommender Systems

Most machine learning models in industry are *supervised*: they predict a value given a set of features (also called inputs) and labels (also called the "ground-truth").  These models can inform us *what* will be the label given a set of features, but they don't tell us *how* to maximize some utility.  We must define a *policy*, which takes in the outputs of these supervised models and decides which items to recommend.

While part of the utility is immediate (e.g. a click or a thumbs-up on a post), there are other facets of the utility that can only be measured over time such as long-term engagement.  RL allows us to maximize these long-term benefits and balance them with the short term benefits effectively.

## RL + Infrastructure

Many problems in infrastructure involve stateful, dynamical systems.  For example, one may want to stream video to a cell phone from a server in the cloud.  The available bandwidth and the amount of the video that has already been buffered can be encoded into a state, and the action can be the quality of the next video segment to request.  As low-quality buffers fill up or the bandwidth increases, we can shift to requesting higher quality video and improve the user experience.


## RL + You

While historically, RL has been primarily used in the context of robotics and game-playing, it can be employed in a variety of problem spaces. At Facebook, we're working on using RL at scale: suggesting people you may know, notifying you about fiend & page updates, optimizing our streaming video bitrate, and more.  If you use BlueWhale for your project, [let us know](contact.html)!
