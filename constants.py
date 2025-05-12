
# UNITS ARE -> kg, m, s, N
# TODO: change all the test and train scripts to use these units

gravity=-9.81
dt = 1 / 100
base_size=(0.01, 0.01)
base_mass=1
link_size=(0.002, 0.15)
link_mass=0.1
groove_length = 0.7
max_steps = 50 // dt
actuation_max = 1 # force 
margin = 1

# # for workshop
# base_size=(10, 9)
# base_mass=1
# link_size=(1, 30)
# link_mass=0.1
# actuation_max=1500 # force or speed

