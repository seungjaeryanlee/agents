# Google Summer of Code 2019 with TensorFlow: Random Network Distillation

This branch is a fork of TF-Agents to showcase my work during Google Summer of Code 2019 with TensorFlow.

To avoid confusion, this branch is kept minimal, removing all unneeded files.

**For a more comprehensive review, [check my blog series!](https://www.endtoend.ai/blog/tags/gsoc/)**

## Installation

First, install necessary PyPI packages:

```bash
pip install tf-nightly tfp-nightly gin-config gym
pip install -e .
```

If you plan to run it on MountainCar (Box2D environment) or Montezuma's Revenge (Atari environment), also download extra OpenAI Gym packages.

```bash
pip install gym[box2d]  # For Box2D environments
pip install gym[atari]  # For Atari environments
```

## Running RND

To reproduce the results on Mountain Car, run the command below:

```bash
python tf_agents/agents/ppo/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/rndppo/gym/LunarLander-v2/ \
  --logtostderr --use_rnd
```

During or after the run, you can visualize the training process through TensorBoard.

```bash
tensorboard --logdir $HOME/tmp/rndppo/gym/LunarLander-v2/ --port 2223
```

