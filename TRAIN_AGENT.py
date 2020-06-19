import os
import sys
import numpy as np
import json

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from LANGEVIN2D_ENV import Langevin2D_Env

###############################################################################
#       PARAMETERS
###############################################################################
# Parallel
num_env = 8

# Saver directory
directory = os.path.join(os.getcwd(), 'agents' ,'saver_data_D_0_dta_0p05_maxa_1_ep100_lstm2_32_dense_64_gr_1_wn_1_r_ma1em2')

# Environment Parameters
env_params = {
    "dt": 0.0005,
    "T" : 100.0,
    "a" : 1.0 + 1.0j,
    "b" : -5.0e2,
    "D" : 0.0e-4,
    "x0": None
    }

# Controller Parameters
optimization_params = {
    "min_value_forcing": -1.0,
    "max_value_forcing": 1.0
    }

# Training Parameters
training_params = {
    "num_episodes" : 400,
    "dt_action"    : 0.05
}

# Compute environment and action input timesteps
n_env_steps = int(training_params["dt_action"] / env_params["dt"])
max_episode_timesteps = int(env_params["T"]/env_params["dt"]/n_env_steps)

# Create and instance of the complex Stuart-Landau environment
environment = Langevin2D_Env(n_env_steps = n_env_steps)
environment.env_params = env_params
environment.optimization_params = optimization_params

environments = []

for env in range(num_env):
    environments.append(Langevin2D_Env(n_env_steps = n_env_steps))
    environments[env].env_params = env_params
    environments[env].optimization_params = optimization_params

###############################################################################
#       ACTOR/CRITIC NETWORK DEFINITIONS
###############################################################################

# Specify network architecture
# DENSE LAYERS
# actor_network = [   
#         dict(type='retrieve', tensors='observation'),
#         dict(type='dense', size=2),
#     ]

# LSTM
actor_network = [
    [   
        dict(type='retrieve', tensors='observation'),
        dict(type='internal_lstm', size=32, length=2, bias=False),
        dict(type='register' , tensor ='intermed-1')
    ],
    [   
        dict(type='retrieve', tensors='prev_action'),
        dict(type='internal_lstm', size=32, length=2, bias=False),
        dict(type='register' , tensor ='intermed-2')
    ],
    [
        dict(type='retrieve', tensors=['intermed-1','intermed-2'], aggregation='concat'),
        dict(type='dense', size=64),
    ]
]

critic_network = actor_network

###############################################################################
#       AGENT DEFINITION
###############################################################################

# Specify the agent parameters - PPO algorithm
agent = Agent.create(
    # Agent + Environment
    agent='ppo',  # Agent specification
    environment=environment,  # Environment object
    exploration=0.0,
    # Network
    network=actor_network,  # Policy NN specification
    # Optimization
    batch_size=num_env,  # Number of episodes per update batch
    learning_rate=1e-4,  # Optimizer learning rate
    subsampling_fraction=0.33,  # Fraction of batch timesteps to subsample
    optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, # The epsilon of the ppo CLI objective
    estimate_terminal=False,  # Whether to estimate the value of terminal states
    # TODO: gae_lambda=0.97 doesn't currently exist - ???
    # Critic
    critic_network=critic_network,  # Critic NN specification
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-4)
    ),
    # Regularization
    entropy_regularization=0.01,  # To discourage policy from being too 'certain'
    # Parallel
    parallel_interactions=num_env,
    # TensorFlow
    saver=dict(directory=directory, filename="agent"),  # TensorFlow saver configuration for periodic implicit saving
    # TensorBoard Summarizer
    #summarizer=dict(directory=os.path.join(directory, 'summarizer') , labels="all")
)

###############################################################################
#       TRAINING
###############################################################################

# Runner definition - Serial runner
runner = Runner(
    environments=environments,
    agent=agent,
    remote="multiprocessing",
    max_episode_timesteps=max_episode_timesteps,
    #evaluation=True
)

# Proceed to training
runner.run(
    num_episodes=training_params["num_episodes"],
    sync_episodes=True,
    #save_best_agent=os.path.join(os.getcwd(), 'best_agent')
)

agent.save()

runner.close()