import numpy as np
import matplotlib.pyplot as plt
from constants import * 

cut_samples_no = 50
obs_data = np.load('data\\obs_log.npy')[:-cut_samples_no,:]
force = np.load('data\\force_log.npy')[:-cut_samples_no]
x = obs_data[:,0]
x_dot = obs_data[:,1]
theta = obs_data[:,2]
theta_dot = obs_data[:,3]

dt = dt # s
num_steps = obs_data.shape[0]  # Number of time steps from obs_data
t = np.arange(0, num_steps * dt, dt)

M = base_mass # kg
m = link_mass # kg

cart_size = np.array(base_size)/100 # m
link_size = np.array(link_size)/100 # m
groove_length = groove_length/100   # m

r_pulley = 18.5e-3   # m, assumed

I = (m*(link_size[1]/2)**2)/3; # pendulum inertia 
J = (I+m*(link_size[1]/2)**2); 

rpm = (x_dot / (2 * np.pi * r_pulley)) * 60 # rpm
torque = force * r_pulley # Nm
omega = (rpm * 2 * np.pi) / 60 # rad/s
power = torque * omega  # W

# print(t.shape, obs_data.shape, force.shape, rpm.shape)

# Create subplots
fig, ax = plt.subplots(8, 1, figsize=(8, 12), sharex=True)

# Plot each observation
ax[0].plot(t, x, label='x [m]')
ax[0].set_ylabel('x [m]')
ax[0].legend()

ax[1].plot(t, x_dot, label='x_dot [m/s]', color='orange')
ax[1].set_ylabel('x_dot [m/s]')
ax[1].legend()

ax[2].plot(t, theta, label='theta [deg]', color='green')
ax[2].set_ylabel('theta [deg]')
ax[2].legend()

ax[3].plot(t, theta_dot, label='theta_dot [deg/s]', color='red')
ax[3].set_ylabel('theta_dot [deg/s]')
ax[3].legend()

ax[4].plot(t, force, label='Force [N]', color='purple')
ax[4].set_ylabel('Force [N]')
ax[4].legend()

ax[5].plot(t, torque, label='Torque [Nm]', color='cyan')
ax[5].set_ylabel('Torque [Nm]')
ax[5].legend()

ax[6].plot(t, rpm, label='Motor Speed [rpm]', color='magenta')
ax[6].set_ylabel('Motor Speed [rpm]')
ax[6].legend()

ax[7].plot(t, power, label='Power Required [W]', color='brown')
ax[7].set_ylabel('Power Required [W]')
ax[7].legend()

ax[-1].set_xlabel('Time [s]')

plt.tight_layout()
fig.savefig("data\\analysis.png", dpi=300, bbox_inches='tight')
plt.show()



