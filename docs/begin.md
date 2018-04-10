---
id: begin
title: BlueWhale: Applied Reinforcement Learning
hide_title: true
sidebar_label: Introduction
---

<br/>
![BlueWhale Logo](../img/logo/logo_no_background.png)
<br/>

How would you teach a robot to balance a pole? Or safely land a space ship? Or even to walk?

Using reinforcement learning (RL), you wouldn't have to teach it how to do any of these things: only what to do. RL formalizes our intuitions about trial and error – agents take actions, experience feedback, and adjust their behavior accordingly.

An agent may start with awful performance: the cart drops the pole immediately; when the space ship careens left, it tilts further; the walker can't take one step without falling. But with experience from exploration and failure, it learns. Soon enough, the agent is behaving in a way you never explicitly told it to, and is achieving the goals you implicitly set forth. It takes actions that optimize for the reward system you designed, often coming up with solutions and employing strategies you hadn't thought of.

While historically, RL has been primarily used in the context of robotics and game-playing, it can be employed in a variety of problem spaces. At Facebook, we're working on using RL at scale: suggesting people you may know, notifying you about fiend & page updates, optimizing our streaming video bitrate, and more.

Advances in RL theory, including the advent of Deep Query Networks and Deep Actor-Critic models, allow us to use function approximation to approach problems with large state and action spaces.  This project, called BlueWhale, contains Deep RL implementations built on [PyTorch](http://pytorch.org/) and [caffe2](http://caffe2.ai/). Internally, we train these models on large databases of episodes, but externally we provide support for running BlueWhale inside [OpenAI Gym](gym.openai.com).  BlueWhale is extremely fast and can train models with 10M+ parameters on billions of rows of data.
