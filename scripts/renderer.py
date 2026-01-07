import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_animation(pendulum, times, states, controls, filename = None):
    """
    Create animation of the pendulum simulation with detailed analysis plots.
    
    Args:
        pendulum: The physics model class
        times: Array of time stamps (N)
        states: Array of states (N x 6)
        controls: Array of control inputs (N-1 or N)
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], hspace=0.3)

    # --- 1. LEFT: Main Animation Plot ---
    ax_anim = fig.add_subplot(gs[:, 0])
    ax_anim.set_xlim(-pendulum.position_limit - 1, pendulum.position_limit + 1)
    ax_anim.set_ylim(-1.5, 1.5)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.set_xlabel('Position (m)')
    ax_anim.set_ylabel('Height (m)')
    ax_anim.set_title('Double Inverted Pendulum Simulation')
    
    # Visual elements
    ax_anim.axvline(-pendulum.position_limit, color='red', ls='--', alpha=0.3)
    ax_anim.axvline(pendulum.position_limit, color='red', ls='--', alpha=0.3)
    ax_anim.axhline(0, color='black', lw=1)
    
    cart_patch = plt.Rectangle((0, 0), 0.2, 0.1, color='blue', zorder=3)
    ax_anim.add_patch(cart_patch)
    
    rod1_line, = ax_anim.plot([], [], 'k-', lw=4, zorder=2)
    rod2_line, = ax_anim.plot([], [], 'r-', lw=4, zorder=2)
    mass1_point, = ax_anim.plot([], [], 'ko', ms=8, zorder=4)
    mass2_point, = ax_anim.plot([], [], 'ro', ms=8, zorder=4)
    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes)

    # --- 2. RIGHT TOP: States (Angles) ---
    ax_states = fig.add_subplot(gs[0, 1])
    ax_states.set_title('Joint Angles')
    ax_states.set_ylabel('Angle (deg)')
    ax_states.grid(True)
    ax_states.axhline(0, color='green', ls='--', alpha=0.5, label='Target')
    
    line_th1, = ax_states.plot([], [], 'b-', label='Theta 1')
    line_th2, = ax_states.plot([], [], 'r-', label='Theta 2')
    ax_states.legend(loc='upper right', fontsize='small')

    # --- 3. RIGHT MIDDLE: Control History ---
    ax_ctrl = fig.add_subplot(gs[1, 1], sharex=ax_states)
    ax_ctrl.set_title('Control Input')
    ax_ctrl.set_ylabel('Force (N)')
    ax_ctrl.grid(True)
    
    line_u, = ax_ctrl.step([], [], 'k-', where='post')

    # --- 4. RIGHT BOTTOM: Energy ---
    ax_energy = fig.add_subplot(gs[2, 1], sharex=ax_states)
    ax_energy.set_title('System Energy')
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Energy (J)')
    ax_energy.grid(True)
    
    line_ek, = ax_energy.plot([], [], 'b-', label='Kinetic', alpha=0.6)
    line_ep, = ax_energy.plot([], [], 'r-', label='Potential', alpha=0.6)
    line_et, = ax_energy.plot([], [], 'g-', label='Total')
    ax_energy.legend(loc='upper right', fontsize='small')

    # --- Data Pre-processing ---
    # Convert angles to degrees for easier reading
    theta1_deg = np.degrees(states[:, 1])
    theta2_deg = np.degrees(states[:, 2])
    
    # Calculate Energies
    kinetic_energies = []
    potential_energies = []
    total_energies = []
    for s in states:
        Ek, Ep = pendulum.energy(s)
        kinetic_energies.append(Ek)
        potential_energies.append(Ep)
        total_energies.append(Ek + Ep)

    # Handle control array length (it might be 1 shorter than states)
    n_frames = len(times)
    u_plot = np.zeros(n_frames)
    if len(controls) < n_frames:
        u_plot[:len(controls)] = controls
        u_plot[len(controls):] = controls[-1] # repeat last
    else:
        u_plot = controls[:n_frames]

    def init():
        return rod1_line, rod2_line, mass1_point, mass2_point, cart_patch, \
               line_th1, line_th2, line_u, line_ek, line_ep, line_et

    def animate(frame):
        # Stop at end
        if frame >= n_frames: return init()
            
        t_curr = times[:frame+1]
        
        # 1. Update Animation
        state = states[frame]
        cart_pos, j1, m1, j2, m2, end = pendulum.get_pendulum_positions(state)
        
        cart_patch.set_x(cart_pos[0] - 0.1) # Center the 0.2 width cart
        cart_patch.set_y(cart_pos[1] - 0.05)
        
        rod1_line.set_data([j1[0], j2[0]], [j1[1], j2[1]])
        rod2_line.set_data([j2[0], end[0]], [j2[1], end[1]])
        mass1_point.set_data([m1[0]], [m1[1]])
        mass2_point.set_data([m2[0]], [m2[1]])
        time_text.set_text(f'Time: {times[frame]:.2f} s')

        # 2. Update Angles
        # line_th1.set_data(t_curr, theta1_deg[:frame+1])
        # line_th2.set_data(t_curr, theta2_deg[:frame+1])
        th1_wrapped = (theta1_deg[:frame+1] + 180) % 360 - 180
        th2_wrapped = (theta2_deg[:frame+1] + 180) % 360 - 180
        line_th1.set_data(t_curr, th1_wrapped)
        line_th2.set_data(t_curr, th2_wrapped)
        
        # 3. Update Control
        line_u.set_data(t_curr, u_plot[:frame+1])
        
        # 4. Update Energy
        line_ek.set_data(t_curr, kinetic_energies[:frame+1])
        line_ep.set_data(t_curr, potential_energies[:frame+1])
        line_et.set_data(t_curr, total_energies[:frame+1])
        
        # Dynamic Scaling for Right plots
        if frame > 0:
            # Angles
            curr_min = min(np.min(th1_wrapped), np.min(th2_wrapped))
            curr_max = max(np.max(th1_wrapped), np.max(th2_wrapped))
            ax_states.set_xlim(0, times[frame])
            ax_states.set_ylim(curr_min - 10, curr_max + 10)
            
            # Controls
            u_min, u_max = np.min(u_plot[:frame+1]), np.max(u_plot[:frame+1])
            margin = (u_max - u_min) * 0.1 + 1.0 # Add margin + min 1.0 pad
            ax_ctrl.set_ylim(u_min - margin, u_max + margin)
            
            # Energy
            e_min = min(min(kinetic_energies[:frame+1]), min(potential_energies[:frame+1]))
            e_max = max(max(kinetic_energies[:frame+1]), max(potential_energies[:frame+1]))
            ax_energy.set_ylim(e_min - 5, e_max + 5)

        return rod1_line, rod2_line, mass1_point, mass2_point, cart_patch, \
               line_th1, line_th2, line_u, line_ek, line_ep, line_et

    # Use interval based on dt to keep roughly real-time (or slower)
    interval_ms = 1000 * (times[1] - times[0])
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, 
        interval=interval_ms, blit=False, repeat=False
    )

    if filename is not None:
        anim.save(filename, writer=animation.PillowWriter(fps=30))
    
    plt.tight_layout()
    plt.show()
    return fig, anim
