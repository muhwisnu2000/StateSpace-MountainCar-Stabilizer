# Mountain Car Simulation With Disturbances, Reset, and Student-Friendly Control Placeholder

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
g = 9.81    # Gravitational constant (m/s^2)
m = 1.0     # Mass of the car (kg)
k = 3.0     # Scaling factor for the slope's curvature

# Shift the mountain profile so the peak is at the center
peak_shift = np.pi / (2 * k)  # Horizontal shift to align peak at x = 0

# Disturbance parameters
disturbance_force = 5.0  # Magnitude of the disturbance
disturbance_interval = 0.5  # Time interval for disturbances

# Restart threshold
frame_restart_threshold = 1.2  # Restart simulation if the car goes out of this range

# Placeholder for a user-defined control system
def control_system(state):
    """
    Placeholder for a control system.
    Parameters:
    - state: [x, v] where x is position and v is velocity.
    Returns:
    - Force applied to the car (N).
    """
    # Example: Students can implement their own control logic here
    F = 0.0  # No control (default)
    return F

# State-space dynamics of the Mountain Car with disturbances
def mountain_car_dynamics(t, state):
    """
    Computes the dynamics of the mountain car with periodic disturbances and a user-defined control system.
    Parameters:
    - t: Current time (seconds).
    - state: [x, v] where x is position and v is velocity.
    Returns:
    - [v, a]: Derivatives of position and velocity.
    """
    x, v = state
    F = control_system(state)  # Get control force from the user-defined system
    a = -g * np.cos(k * (x + peak_shift)) + F / m  # Acceleration with control

    # Add disturbance every disturbance_interval seconds
    if int(t) % disturbance_interval == 0 and int(t * 100) % 100 == 0:
        a += disturbance_force / m  # Add a disturbance force

    return [v, a]

# Function to reset the simulation if the car moves out of frame
def reset_simulation():
    global current_state, solution_time, solution_states
    current_state = initial_state.copy()
    solution_time = [0]
    solution_states = [current_state]

# Initial conditions
initial_state = [np.random.uniform(0.1,1), 0.0]  # Starting near the peak with no velocity
current_state = initial_state.copy()
solution_time = [0]
solution_states = [current_state]

# Solver function to update the state
def update_solution():
    global current_state, solution_time, solution_states
    t_last = solution_time[-1]
    # Solve for the next time step
    sol = solve_ivp(mountain_car_dynamics, [t_last, t_last + dt], current_state, t_eval=[t_last + dt])
    solution_time.append(sol.t[-1])
    solution_states.append(sol.y[:, -1])
    current_state = sol.y[:, -1]
    # Check if the car goes out of frame
    if abs(current_state[0]) > frame_restart_threshold:
        reset_simulation()

# Simulation parameters
t_span = (0, 10)  # Simulate for 10 seconds
dt = 0.01         # Time step

# Visualization: Animate the Mountain Car
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("Position (x)")
ax.set_ylabel("Height")
ax.set_title("Mountain Car With Disturbances and Reset (Student Control Placeholder)")

# Mountain and Car representation
mountain_x = np.linspace(-1.2, 1.2, 500)
mountain_y = np.sin(k * (mountain_x + peak_shift)) / k  # Adjusted mountain profile
car, = ax.plot([], [], 'ro', markersize=8)
mountain_line, = ax.plot(mountain_x, mountain_y, 'k-', lw=2)

# Initialize the animation
def init():
    car.set_data([], [])
    return car, mountain_line

# Update function for the animation
def update(frame):
    update_solution()
    car_x = current_state[0]
    car_y = np.sin(k * (car_x + peak_shift)) / k  # Compute height from the adjusted mountain profile
    car.set_data([car_x], [car_y])
    return car, mountain_line

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, blit=True, interval=dt * 1000)

plt.show()
