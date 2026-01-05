import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_animation(pendulum, times, states):
    """Create animation of the pendulum simulation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set up pendulum plot
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1.0, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Double Inverted Pendulum')
    
    # Add position limits
    ax1.axvline(-pendulum.position_limit, color='red', linestyle='--', alpha=0.5, label='Position Limits')
    ax1.axvline(pendulum.position_limit, color='red', linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Initialize plot elements
    cart_patch = plt.Rectangle((0, 0), 0.2, 0.1, color='blue')
    ax1.add_patch(cart_patch)
    
    rod1_line, = ax1.plot([], [], 'k-', linewidth=4, label='Rod 1')
    rod2_line, = ax1.plot([], [], 'r-', linewidth=4, label='Rod 2')
    mass1_point, = ax1.plot([], [], 'ko', markersize=8)
    mass2_point, = ax1.plot([], [], 'ro', markersize=8)
    
    # Set up energy plot
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('System Energy')
    ax2.grid(True)
    
    kinetic_line, = ax2.plot([], [], 'b-', label='Kinetic Energy')
    potential_line, = ax2.plot([], [], 'r-', label='Potential Energy')
    total_line, = ax2.plot([], [], 'g-', label='Total Energy')
    ax2.legend()
    
    # Energy data
    kinetic_energies = []
    potential_energies = []
    total_energies = []
    
    for state in states:
        E_kin, E_pot = pendulum.energy(state)
        kinetic_energies.append(E_kin)
        potential_energies.append(E_pot)
        total_energies.append(E_kin + E_pot)
    
    def animate(frame):
        if frame >= len(states):
            return
            
        state = states[frame]
        
        # Get positions
        cart_pos, joint1_pos, mass1_pos, joint2_pos, mass2_pos, end_pos = \
            pendulum.get_pendulum_positions(state)
        
        # Update cart
        cart_patch.set_x(cart_pos[0] - 0.1)
        cart_patch.set_y(cart_pos[1] - 0.05)
        
        # Update rods
        rod1_line.set_data([joint1_pos[0], joint2_pos[0]], 
                          [joint1_pos[1], joint2_pos[1]])
        rod2_line.set_data([joint2_pos[0], end_pos[0]], 
                          [joint2_pos[1], end_pos[1]])
        
        # Update masses
        mass1_point.set_data([mass1_pos[0]], [mass1_pos[1]])
        mass2_point.set_data([mass2_pos[0]], [mass2_pos[1]])
        
        # Update energy plot
        current_time = times[:frame+1]
        kinetic_line.set_data(current_time, kinetic_energies[:frame+1])
        potential_line.set_data(current_time, potential_energies[:frame+1])
        total_line.set_data(current_time, total_energies[:frame+1])
        
        if frame > 0:
            ax2.set_xlim(0, times[frame])
            ax2.set_ylim(min(min(kinetic_energies[:frame+1]), 
                            min(potential_energies[:frame+1])) - 1,
                        max(max(kinetic_energies[:frame+1]), 
                            max(potential_energies[:frame+1])) + 1)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(states), 
                                  interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()

    return fig, anim
