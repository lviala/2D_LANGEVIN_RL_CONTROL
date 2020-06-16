from LANGEVIN2D_ENV import Langevin2D_Env

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Environment Parameters
env_params = {
                "dt": 0.0005,
                "T" : 10.0,
                "a" : 1.0 +1.0j,
                "b" : -5.0e2,
                "D" : 0.0e-4,
                "x0": 0.03 + 0.0j
                }

# Path to save the figure
#fig_path = 'figures/FeedbackControl_Kp5_mag_1_D_0_gr_10_wn_10.png'
#fig_path = 'figures/NoControl_D_1em4_gr_10_wn_10.png'
fig_path = None


# Create instance of complex Stuart-Landau equation environment
environment = Langevin2D_Env()
environment.env_params = env_params

# Initiate environment to initial state
time  = np.zeros((environment.max_episode_timesteps()))
states  = np.zeros((environment.max_episode_timesteps(),2))
actions  = np.zeros((environment.max_episode_timesteps(),2))
state = environment.reset()
states[0,:] = state["observation"]

# Episode reward - defined as magnitude of the complex state
sum_rewards = 0.0

# Set up control time with reference to simulation time
dt = environment.env_params["dt"]
dt_action = 0.05
T = environment.env_params["T"]
n_env_steps = int(dt_action / dt)
n_actions = int(T/dt/n_env_steps)

# Proportional gain - If using feedback control
Kp_r = 0.0
Kp_i = 0.0
max_forcing_mag = 1.0

observation = states[0,:]

# March system for specified number of timesteps

for ii in range(0,n_actions):

    p_control = np.array([-Kp_r*observation[0] , -Kp_i*observation[1]])
    
    for jj in range(0,n_env_steps):
        actions[jj + ii*n_env_steps,:] = np.clip(p_control, -max_forcing_mag, max_forcing_mag)
        state, terminal, reward = environment.execute(actions= p_control)
        observation = state["observation"]
        states[jj + ii*n_env_steps,:] = observation
        time[jj + ii*n_env_steps] = environment.time
        sum_rewards += reward

# Compute and output episode metrics
print('Episode cumulative reward: {} - Average reward: {}'.format(sum_rewards, sum_rewards/environment.max_episode_timesteps()))

fig = plt.figure(figsize=(16,9))
fig.tight_layout()

if (Kp_i == 0 and Kp_r == 0):
    fig.suptitle('No Control - Episode cumulative reward: {} - Average reward: {}'.format(sum_rewards, sum_rewards/environment.max_episode_timesteps()))
else:
    fig.suptitle('Proportional Feedback Control Kp_r={}, Kp_i={} - Episode cumulative reward: {} - Average reward: {}'.format(Kp_r, Kp_i, sum_rewards, sum_rewards/environment.max_episode_timesteps()))

plt.subplots_adjust(top=0.925, bottom=0.05, right=0.95, left=0.05, hspace=0.5)

# 2D Histogram (PDF) of the state
nbins = 200
N_2D, x_edges, y_edges = np.histogram2d(states[:,0],states[:,1], np.array([nbins,2*nbins]))
PDF_2D = N_2D / environment.max_episode_timesteps() / (x_edges[1]-x_edges[0]) / (y_edges[1]-y_edges[0])

# Plot 2D PDF as pcolormesh
X,Y = np.meshgrid(x_edges, y_edges)
ax0 = plt.subplot2grid(shape=(4,2), loc=(0,0), rowspan=2, colspan= 1)
im = ax0.pcolormesh(X, Y, PDF_2D.T, cmap= plt.get_cmap('hot_r'))
fig.colorbar(im, ax = ax0)
ax0.set_title('2D PDF of the system states')
ax0.set_xlabel('Re(x)')
ax0.set_ylabel('Im(x)')

# 1D PDF
N_1D_re , x_edges_1D_re = np.histogram(states[:,0],bins = nbins)
PDF_1D_re = N_1D_re / environment.max_episode_timesteps() / (x_edges_1D_re[1] - x_edges_1D_re[0])

N_1D_im , x_edges_1D_im = np.histogram(states[:,1],bins = nbins)
PDF_1D_im = N_1D_im / environment.max_episode_timesteps() / (x_edges_1D_im[1] - x_edges_1D_im[0])
# Plot 1D PDF
ax1 = plt.subplot2grid(shape=(4,2), loc=(2,0), rowspan=1, colspan= 1)
ax1.plot(x_edges_1D_re[:-1], PDF_1D_re)
ax1.set_title('1D PDF of the real component of the state')
ax1.set_xlabel('Re(x)')
ax1.set_ylabel('P(Re(x))')

ax2 = plt.subplot2grid(shape=(4,2), loc=(3,0), rowspan=1, colspan= 1)
ax2.plot(x_edges_1D_im[:-1], PDF_1D_im)
ax2.set_title('1D PDF of the imaginary component of the state')
ax2.set_xlabel('Im(x)')
ax2.set_ylabel('P(Im(x))')

# Estimate power spectral density using Welch method
n_window = int(environment.max_episode_timesteps()/10)
Fs = 1/environment.env_params["dt"]
window = signal.get_window('hann', n_window)
f_re , PSD_re = signal.welch(states[:,0], fs= Fs, window= window, noverlap= 0.5*n_window, nfft= n_window)
f_im , PSD_im = signal.welch(states[:,1], fs= Fs, window= window, noverlap= 0.5*n_window, nfft= n_window)
# Plot PSD
ax3 = plt.subplot2grid(shape=(4,2), loc=(0,1), rowspan=2, colspan= 1)
ax3.loglog(f_re, PSD_re)
ax3.loglog(f_im, PSD_im)
ax3.set_title('Power Spectral Density of the state')
ax3.set_xlabel('f [Hz]')
ax3.set_ylabel('S(x)')
ax3.legend(('Real', 'Imaginary'))

# Plot trajectory of system
ax4 = plt.subplot2grid(shape=(4,2), loc=(2,1), rowspan=1, colspan= 1)
ax4.plot(time, states[:,0])
ax4.plot(time, states[:,1])
ax4.set_title('Trajectory of the state')
ax4.set_xlabel('t')
ax4.set_ylabel('x')
ax4.legend(('Real', 'Imaginary'))

# Plot control input to the system
ax5 = plt.subplot2grid(shape=(4,2), loc=(3,1), rowspan=1, colspan= 1)
ax5.plot(time, actions[:,0])
ax5.plot(time, actions[:,1])
ax5.set_title('Control input to the system - Controller time rate: {}'.format(dt_action))
ax5.set_xlabel('t')
ax5.set_ylabel('u')
ax5.legend(('Real', 'Imaginary'))

if fig_path is not None:
    fig.savefig(fig_path)
plt.show()
