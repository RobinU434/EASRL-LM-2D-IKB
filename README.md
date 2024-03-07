
# EASRL-LM:2D-IKB - Expanding Action Space in Reinforcement Learning through Latent Models: A 2D Inverse Kinematics Benchmark

## Description:

This repository contains the code and data for my Bachelor's thesis, titled "Expanding Action Space in Reinforcement Learning through Latent Models: A 2D Inverse Kinematics Benchmark". In this work, I investigated methods to expand the possible action space dimension in reinforcement learning using dimensionality reduction techniques with normal feed-forward networks, VAEs (Variational Autoencoders), and CVAEs (Conditional VAEs).

The experiments were conducted on a 2D inverse kinematics environment to allow for easy scaling of the action space without significantly altering the task itself. Different task variations were explored, including reaching a specific goal, imitating a known solver's behavior, and hybrid combinations of these tasks.

## Authors:

- Robin Uhrich
- Supervisor: Jasper Hoffmann

## License:
MIT License

## Installation:

This project uses Poetry for dependency management. To install the required dependencies, run the following command in the project directory:

```Bash
poetry install --no-root
```

## Usage:

The project utilizes the rl package for running experiments. You can access the help message for available commands and options by running:

```Bash
python -m rl --help  # for reinforcement learning
python -m latent --help  # for latent model learning
```

Further instructions and guidance can be found within the help messages provided by the `rl` and `latent` package.

The environment is a dedicated submodule and can be found [here](https://github.com/RobinU434/IK-RL). Please Note that the main branch of this module has diverged from the source code needed for this environment. But there is a dedicated branch called `bachelor_thesis` in [`IK-RL`](https://github.com/RobinU434/IK-RL) containing the code compatible with this repo.

## Configuration:

Experiment hyperparameters are primarily defined within the configuration files located in the project directory. These files provide a centralized location for managing and modifying experiment settings.

## Data:

While data is typically generated directly from the environment during training, the script create_complete_dataset.sh allows you to generate a custom dataset for supervised and semi-supervised learning approaches.

## Documentation

If you are interested in the final thesis you can find it [here](./documentation/thesis/thesis_main.pdf). Please not that the access to the original data can't be guaranteed in the future due to its size.   
Code documentation  can be easily created with [`PyDoxyUML`](https://github.com/RobinU434/PyDoxyUML). 

I hope this README provides a clear and comprehensive overview of the project. Feel free to explore the code and experiment with different configurations!
