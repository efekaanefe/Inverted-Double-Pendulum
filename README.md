# Inverted-Double-Pendulum
This project simulates a single inverted pendulum using Pymunk, a 2D physics engine for Python. The goal is to balance the pendulum by applying a force to its base within a given range, calculated by various control methods. The simulation was developed as part of a graduation project at METU (Middle East Technical University) and serves as a testbed for experimenting with NEAT, PID, and LQR controllers. Additionally, PID and LQR were implemented on a real physical system.


## Known Issues
- No Friction: The simulation currently lacks friction, reducing its realism compared to the physical system.
- Speed Control Glitches: Setting the base's speed as the control action (instead of applying a force) causes the base to leave the screen at the boundaries, resulting in glitches.

## Future Work
- Add friction to the simulation to better reflect real-world physics.
- Resolve boundary issues when using speed as the control action, ensuring the base remains within the simulation window.
- Expand the project by further exploring reinforcement learning or other control methods.
