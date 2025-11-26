# Full-State Compensator

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter sistem
g = 9.81    # Konstanta gravitasi (m/s^2)
m = 1.0     # Massa mobil (kg)
k = 3.0     # Faktor skala untuk kelengkungan lereng

# Geser profil gunung agar puncaknya berada di tengah
peak_shift = np.pi / (2 * k)  # Geseran horizontal untuk menyelaraskan puncak di x = 0

# Parameter gangguan (disturbance)
# disturbance_force = 5.0     # Besarnya gangguan
# disturbance_interval = 0.5  # Interval waktu untuk gangguan

# Ambang batas restart
# frame_restart_threshold = 1.2  # Restart simulasi jika mobil keluar dari jangkauan ini

# Linearized Model Matrices (A, B, C)
# A = [[0, 1], [g*k, 0]]
A = np.array([[0, 1], 
              [g * k, 0]])
B = np.array([[0], 
              [1/m]])
C = np.array([[1, 0]]) # Diasumsikan hanya mengukur posisi (x)

### Pole Placement ###
# Poles Controller: Harus negatif agar stabil.
# Poles Observer: Harus lebih negatif (lebih cepat) dari Controller.

poles_controller = [-6, -7]       # Desain respon sistem
poles_observer   = [-30, -31]     # Desain kecepatan estimasi

# Hitung Gain K (Controller) 
# place_poles menghitung K untuk (A - BK)
res_k = place_poles(A, B, poles_controller)
K = res_k.gain_matrix
print(f"Controller Gain K: {K}")

# Hitung Gain L (Observer) 
# Kita gunakan transpose karena place_poles menyelesaikan A - BK, sedangkan observer A - LC
res_l = place_poles(A.T, C.T, poles_observer)
L = res_l.gain_matrix.T
print(f"Observer Gain L: {L}")

### Implementasi Compensator ###

# Variabel Global untuk menyimpan state estimasi (x_hat, v_hat)
# Inisialisasi estimasi di 0
observer_state = np.array([0.0, 0.0]) 
last_t = 0

# Placeholder untuk sistem kendali yang didefinisikan pengguna
def control_system(t, y_measurement):
    # Target (Puncak gunung & Diam)
    desired_state = np.array([0.0, 0.0]) 
    
    # Hitung Error (State Estimasi - Target)
    error = observer_state - desired_state
    
    # u = -K * error
    u = -np.dot(K, error)

    return u[0]

# Dinamika state-space Mountain Car
def mountain_car_dynamics(t, state):
    """
    Menghitung dinamika mountain car dengan gangguan periodik dan sistem kendali yang didefinisikan pengguna.
    Parameter:
    - t: Waktu saat ini (detik).
    - state: [x, v] di mana x adalah posisi dan v adalah kecepatan.
    Kembalian:
    - [v, a]: Turunan posisi dan kecepatan.
    """
    x, v = state
    F = control_system(t, x)  # Dapatkan gaya kendali dari sistem yang didefinisikan pengguna
    a = -g * np.cos(k * (x + peak_shift)) + F / m  # Percepatan dengan kendali

    # Tambahkan gangguan setiap disturbance_interval detik
    # if int(t) % disturbance_interval == 0 and int(t * 100) % 100 == 0:
    #     a += disturbance_force / m  # Tambahkan gaya gangguan

    return [v, a]

# Fungsi untuk mereset simulasi jika mobil bergerak keluar frame
# def reset_simulation():
#     global current_state, solution_time, solution_states
#     current_state = initial_state.copy()
#     solution_time = [0]
#     solution_states = [current_state]
# Tidak diperlukan karena mobil tidak akan bergerak keluar frame (Kendali sudah diterapkan)

# Kondisi awal
initial_state = [np.random.uniform(-0.5, 0.5), 0.0] # Mulai acak
current_state = initial_state.copy()
observer_state = np.array([0.0, 0.0]) # Reset observer
last_t = 0
solution_time = [0]
solution_states = [current_state]
solution_estimated = [observer_state.copy()]

# Fungsi solver untuk memperbarui state
def update_solution():
    global current_state, solution_time, solution_states, last_t, observer_state
    
    t_curr = solution_time[-1]
    
    # --- Update Observer ---
    # Ambil data yang dibutuhkan
    y_measurement = current_state[0]                # Posisi asli (y)
    u_val = control_system(t_curr, y_measurement)   # Gaya yang diberikan (u)
    
    # Rumus Observer: d_x_hat = A*x_hat + B*u + L(y - C*x_hat)
    
    # Hitung komponen rumus
    y_hat = np.dot(C, observer_state)         # C * x_hat
    error = y_measurement - y_hat             # (y - y_hat)
    
    term1 = np.dot(A, observer_state)         # A * x_hat
    term2 = np.dot(B, [u_val]).flatten()      # B * u
    term3 = np.dot(L, error).flatten()        # L * error
    
    d_x_hat = term1 + term2 + term3           # Total Turunan
    
    # Update State Observer (x_hat baru = x_hat lama + turunan * dt)
    observer_state = observer_state + d_x_hat * dt
    
    # --- Update Plant (Mobil) ---
    sol = solve_ivp(mountain_car_dynamics, [t_curr, t_curr + dt], current_state, t_eval=[t_curr + dt])
    
    if sol.y.size > 0:
        current_state = sol.y[:, -1]
        solution_time.append(sol.t[-1])
        solution_states.append(current_state)
        solution_estimated.append(observer_state.copy()) # Simpan log estimasi

# Parameter simulasi
t_span = (0, 15)  # Simulasikan selama 15 detik
dt = 0.01         # Langkah waktu

# --- 5. VISUALISASI ---
fig = plt.figure(figsize=(10, 8))
fig.suptitle("Challenge 1: FSC (Default Backend)")

# Subplot 1: Animasi
ax_anim = fig.add_subplot(2, 1, 1)
ax_anim.set_xlim(-1.2, 1.2)
ax_anim.set_ylim(-1.0, 1.0)
ax_anim.set_ylabel("Height")
ax_anim.set_title("Simulation Animation")

# Subplot 2: Posisi
ax_pos = fig.add_subplot(2, 2, 3)
ax_pos.set_xlim(0, 15)
ax_pos.set_ylim(-1.0, 1.0)
ax_pos.set_xlabel("Time (s)")
ax_pos.set_ylabel("Position (m)")
ax_pos.grid(True)

# Subplot 3: Kecepatan
ax_vel = fig.add_subplot(2, 2, 4)
ax_vel.set_xlim(0, 15)
ax_vel.set_ylim(-3, 3)
ax_vel.set_xlabel("Time (s)")
ax_vel.set_ylabel("Velocity (m/s)")
ax_vel.grid(True)

# Aset Gambar
mountain_x = np.linspace(-1.2, 1.2, 500)
mountain_y = np.sin(k * (mountain_x + peak_shift)) / k
ax_anim.plot(mountain_x, mountain_y, 'k-', lw=2)
car, = ax_anim.plot([], [], 'ro', markersize=10, label='Real Car')
est_car, = ax_anim.plot([], [], 'bx', markersize=8, alpha=0.5, label='Estimator')
ax_anim.legend()

line_pos, = ax_pos.plot([], [], 'b-', lw=2, label='True Position')
ax_pos.plot([0, 15], [0, 0], 'r--', alpha=0.7, label='Target')

line_true_vel, = ax_vel.plot([], [], 'g-', lw=2, label='True Velocity')
line_est_vel, = ax_vel.plot([], [], 'r--', lw=2, label='Est Velocity')
ax_vel.legend()

def init():
    car.set_data([], [])
    est_car.set_data([], [])
    line_pos.set_data([], [])
    line_true_vel.set_data([], [])
    line_est_vel.set_data([], [])
    return car, est_car, line_pos, line_true_vel, line_est_vel

def update(frame):
    update_solution()
    
    car_x = current_state[0]
    car_y = np.sin(k * (car_x + peak_shift)) / k
    car.set_data([car_x], [car_y])
    
    est_x = solution_estimated[-1][0]
    est_y = np.sin(k * (est_x + peak_shift)) / k
    est_car.set_data([est_x], [est_y])
    
    times = solution_time
    states_arr = np.array(solution_states)
    est_arr = np.array(solution_estimated)
    
    if len(states_arr) > 1:
        line_pos.set_data(times, states_arr[:, 0])
        line_true_vel.set_data(times, states_arr[:, 1])
        line_est_vel.set_data(times, est_arr[:, 1])
    
    return car, est_car, line_pos, line_true_vel, line_est_vel

ani = animation.FuncAnimation(fig, update, frames=750, init_func=init, blit=False, interval=20)

print("Membuka window grafik...")

plt.show()