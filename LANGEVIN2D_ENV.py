import numpy as np
from tensorforce.environments import Environment

class Langevin2D_Env(Environment):

    '''
    Defines the parameters for the 2D Langevin Stuart Landau equation
    dt (float)          - time step for dynamics evolution - default: 0.0005
    T (float)           - total simulation time - default: 100.0
    a (complex float)   - Re(a): growth rate of the dynamics - Im(a): angular frequency at equilibrium - default: 10.0 +0.0j
    b (complex float)   - Re(b): saturation term - Im(b): strength of the non-linear coupling between amplitude and frequency - default: -5.0e2
    D (float)           - Diffusion coefficient associated with Gaussian White Noise forcing - default: 1.0e-2
    x0 (Complex float)  - Initial position of the system - default: 0.03 + 0.0j
    '''
    env_params = {
                "dt": 0.0005,
                "T" : 100.0,
                "a" : 1.0 +1.0j,
                "b" : -5.0e2,
                "D" : 1.0e-1,
                "x0": 0.03 + 0.0j
                }
     
    optimization_params = {
                        "min_value_forcing": -1.0,
                        "max_value_forcing": 1.0
                        }


    def __init__(self, n_env_steps = 1):
        super().__init__()
        self.state = 0.0 + 0.0j # Internal state of the syste,
        self.time  = 0.0        # Internal time of the system
        self.n = 0              # Step number
        self.N = int(self.env_params["T"] / self.env_params["dt"])  # Maximum number of steps to take
        self.n_env_steps = n_env_steps  # Number of environment steps to march the system between actions
        print(self.N)

    def states(self):
        '''
        Returns the state space specification.
        :return: dictionary of state descriptions with the following attributes:
        type ("bool" / "int" / "float") – state data type (required)
        shape (int > 0 / iter[int > 0]) – state shape (default: scalar)
        '''
        return dict(observation = dict(type='float', shape=(2,)), prev_action = dict(type='float', shape=(2,)))
        
    def actions(self):
        '''
        Returns the action space specification.
        :return: dictionary of action descriptions with the following attributes:
        type ("bool" / "int" / "float") – action data type (required)
        shape (int > 0 / iter[int > 0]) – action shape
        min_value/max_value (float) – minimum/maximum action value
        '''
        return dict(type='float', shape=(2,),
                    min_value=self.optimization_params["min_value_forcing"],
                    max_value=self.optimization_params["max_value_forcing"])

    def reset(self):
        """
        Reset environment and setup for new episode.
        Returns:
            initial state of reset environment.
        """
        # Reset simulation time
        self.time = 0.0
        self.n = 0

        # Reset environment to initial position
        if self.env_params["x0"] is not None:
            self.state = self.env_params["x0"]
        else:
            # Initial position on limit-cycle
            eq = np.sqrt(-np.real(self.env_params["a"])/np.real(self.env_params["b"]))
            self.state = eq*np.exp(np.random.normal(scale= 0.5*np.pi)*1j)
        
        print(self.state)
        
        self.N = int(self.env_params["T"] / self.env_params["dt"])  # Maximum number of steps to take
        
        next_state = dict(
            observation = np.array([np.real(self.state),np.imag(self.state)]).flatten(),
            prev_action = np.zeros((2,))
            )

        return(next_state)
    
    def execute(self, actions = np.array([0.0,0.0])):
        '''
        Run solver for one action step, until next RL env state (this means to run for number_steps_execution)
        :param: actions
        :return: next state (state value at end of action step)
                 terminal
                 reward (magnitude of the state)
        '''

        action = actions[0] + actions[1]*1j
        
        # Parameters of the system
        a = self.env_params["a"]
        b = self.env_params["b"]
        D = self.env_params["D"]

        # Solver parameters
        dt = self.env_params["dt"]

        # Gaussian White Noise forcing
        sigma = np.sqrt(2*D) # STD of stochastic forcing
        
        # March the system using Euler-Maruyama method for discretization
        # The system will evolve by n_env_steps steps between control input
        cum_reward = 0.0

        for _ in range(self.n_env_steps):
            An = np.random.normal(0.0,sigma) + np.random.normal(0.0,sigma)*1j

            # Deterministic component of system: complex Stuart-Landau equation
            SL_deterministic = a * self.state + b * self.state * np.square(np.abs(self.state))
            
            self.time = self.time + dt
            self.state = self.state + SL_deterministic * dt + action * dt + An * np.sqrt(dt)

            self.n += 1
            cum_reward -= np.abs(self.state)

        # Extract Real and Imaginary part of state as two separate states
        # Ensure reshape to size (2,)
        next_state = dict(
            observation = np.array([np.real(self.state),np.imag(self.state)]).reshape(2,),
            prev_action = actions
            )

        terminal = False

        # Reward based on magnitude of the state
        #reward = -np.abs(self.state)
        
        # Reward based on average magnitude of the state
        #reward =cum_reward / self.n_env_steps

        # Reward based on average magnitude of the state and action input penalization
        reward = cum_reward / self.n_env_steps - 1e-2*(np.abs(action) / self.optimization_params["max_value_forcing"])

        # Print completion status to console
        if (self.n % (self.N/20) == 0):
            print(self.n)

        return (next_state, terminal, reward)

    def max_episode_timesteps(self):
        
        N = int(self.env_params["T"] / self.env_params["dt"])
        return N


if __name__ == "__main__": 
    env = Langevin2D_Env()

    next_state = env.reset()
    print(next_state)

    next_state, terminal, reward = env.execute()
    print(next_state, terminal, reward)