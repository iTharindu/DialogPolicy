# Dialogue Policy Optimization in Low reseource Settings using Reinforcement Learning

*An implementation of the paper "Dialog policy optimization for low resource setting using Self-play and
Reward based Sampling"*

This document describe how to run the policy optimization using different RL algorithm using the Self-play approach and Reward Based Sampling

**Note** : We have used the user simulator described in [A User Simulator for Task-Completion Dialogues](http://arxiv.org/abs/1612.05688) as the simulator. Github link to the user simulator can be found on [here](https://github.com/MiuLab/TC-Bot).

## Content
* [Introduction](#introduction)
* [Running Dialogue Agents](#running-dialogue-agents)
* [Running Experiment](#running-the-experiment)
* [Results](#evaluation)
* [Reference](#reference)

## Introduction

The dialogue policy optimization is open-research problem and currently the state of the art methods have been based on Reinforcement Learning(RL). However, RL based methods tend to overfit in low resource setting. We are introducing a novel probability based method to address the overfitting problem. Since this methodology applies to a low amount of samples, this method can lead to an insufficient exploration of agendas by the agent. Therefore we further developed the methodology by introducing a selective sampling method based on the reward function that prioritizes the failed dialog acts, where the agent actively decides what agendas to use.

### Architecture Diagram

Diagram below shows the overall achitecture of the system

![Architecture Diagram](imgs/architecture_diagram.png)

### Directory Structure
The important files and directories of the repository is shown below

    ├── src
        ├── deep_dialog
            ├── agents : contain RL agents
            ├── data : dialogue dataset with knowledge base and goals
            ├── dialog_system : contain dialogue manager
            ├── models : nlu and nlg models
            ├── nlg : Natural Language Generation Module
            ├── nlu : Natural Language Undersatanding Module 
            ├── qlearning : qlearning implementaiton in numpy
            ├── self_play : Self Play algorithm with reward based sampling
            ├── usersims : Agenda based user simualator
        ├── run_RL_experiment.py : To run the RL experiment on a given RL algorithm
        ├── run_RL.py : To run RL agent with the self-play and reward based sampling 
        ├── run.py : run agent without self-play and reward based sampling
        ├── user_goal_handler.py : run self-play and reward based sampling experiment varying number of samples
        ├── config.json : configuration file for user_goal_handler
        └── config_rl.json : configuration file for run_RL_experiment

### Requirements 

The system is implemented using Python version 2(Since the simualtor we used is implemented using Python 2)

The following requirements are required
```commandline
numpy
pandas
PyTorch
collections
typing
```
## Running Dialogue Agents

### Rule Agent
```sh
python run.py --agt 5 --usr 1 --max_turn 40
	      --episodes 150
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
```

### Cmd Agent
NL Input
```sh
python run.py --agt 0 --usr 1 --max_turn 40
	      --episodes 150
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
	      --run_mode 0
	      --cmd_input_mode 0
```
Dia_Act Input
```sh
python run.py --agt 0 --usr 1 --max_turn 40
	      --episodes 150
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p 
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
	      --run_mode 0
	      --cmd_input_mode 1
```

### End2End RL Agent
Train End2End RL Agent without NLU and NLG (with simulated noise in NLU)
```sh
python run.py --agt 9 --usr 1 --max_turn 40
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 500
	      --simulation_epoch_size 100
	      --write_model_dir ./deep_dialog/checkpoints/rl_agent/
	      --run_mode 3
	      --act_level 0
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --warm_start 1
	      --warm_start_epochs 120
```
Train End2End RL Agent with NLU and NLG
```sh
python run.py --agt 9 --usr 1 --max_turn 40
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 500
	      --simulation_epoch_size 100
	      --write_model_dir ./deep_dialog/checkpoints/rl_agent/
	      --run_mode 3
	      --act_level 1
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --warm_start 1
	      --warm_start_epochs 120
```

## Running Experiment

### Experiment 01 : To compare our approach with a baseline

The experiment 01 is run to measure the performance of Self-play approach against a baseline model. So We have run the experiment with and without Self-Play as well as with and without Reward based smpling.

The experiment is run varying number of training samples. To setup a minimum and maximum number of samples edit user_goal_hendler.py

Also the configuration file for running experiment 01 is config.json

To run the experiment 

```commandline
python user_goal_handler.py
```
### Experiment 02 : To measure the rate of convergence

The experiment 02 is conducted to measure the rate of convergence of the Self-play approach and to check whether there is apparent lag between training success rate and test success rate

The configuration file for running experiment 01 is config_rl.json

```commandline
python run_RL_experiments.py
```

***The results from experiments are shown in the Results Section***

## Results

For both experiments the dotted lines indicate training success rate while the solid line indicate the test success rate

### Experiment 01
![Experiment 01](imgs/plot_01.png)

### Experiment 02
![Experiment 02](imgs/plot_02.png)

## Reference

Main papers to be cited

```
@inproceedings{Tharindu2020Dialog,
  title={Dialog policy optimization for low resource setting using Self-play and Reward based Sampling},
  author={Tharindu Madusanka, Durashi Langappuli, Thisara Welmilla Uthayasanker Thayasivam and Sanath Jayasena},
  booktitle={34th Pacific Asia Conference on Language, Information and Computation},
  year={2020}
}
```