# 2D_LANGEVIN_RL_CONTROL
Reinforcement Learning based control of the stochastic Stuart-Landau equation

# Installation
[TensorForce](https://github.com/tensorforce) version 0.5.4 is required to run the environment and train the RL agent.

# Training an Agent
In TRAIN_AGENT.py, define:

- the directory for the TensorForce model checkpointing:

```python
# Saver directory
directory = os.path.join(os.getcwd(), 'agents' ,'saver_data_model_name')
```

- the environment parameters for the Stuart-Landau system:
``` python
# Environment Parameters
env_params = {
    "dt": 0.0005,
    "T" : 100.0,
    "a" : 1.0 + 1.0j,
    "b" : -5.0e2,
    "D" : 0.0e-4,
    "x0": 0.03 + 0.0j
    }

# Controller Parameters
optimization_params = {
    "min_value_forcing": -1.0,
    "max_value_forcing": 1.0
    }
    
# Training Parameters
training_params = {
    "num_episodes" : 300,
    "dt_action"    : 0.05
}
```
    
- the actor and critic Neural Networks as a list of layer lists (refer to the [TensorForce](https://tensorforce.readthedocs.io/en/latest/index.html) documentation). The following example integrates dense and LSTM layers: 
```python
network = [
    [   
        dict(type='retrieve', tensors='observation'),
        dict(type='dense', size=16),
        dict(type='register' , tensor ='intermed-1')
    ],
    [   
        dict(type='retrieve', tensors='prev_action'),
        dict(type='dense', size=16),
        dict(type='register' , tensor ='intermed-2')
    ],
    [
        dict(type='retrieve', tensors=['intermed-1','intermed-2'], aggregation='concat'),
        dict(type='internal_lstm', size=32, length=1, bias=True),
        dict(type='dense', size=16),
    ]
]
```

- if required, the additional Agent and runner parameters

- execute the script and training will execute.

# Evaluating an agent
In EVAL_ENV_RL, define:
- the environment parameters, in a similar way as shown above
- the path and filename of the figure to be saved. Define 'figpath=None' if figure is not to be saved:
```python
# Path to save the figure
fig_path = 'figures/RLControl_Run_Description.png'
```

- the path to the TensorForce model saver:

```python
# Saver directory
directory = os.path.join(os.getcwd(), 'agents' ,'saver_data_model_name')
```
- execute the script.
