# Reduced-Order Compensator

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

### Partisi Matriks untuk Reduces Order ###

# A11 = 0, A12 = 1 (Hubungan pos ke vel)
# A21 = gk, A22 = 0 (Hubungan vel ke pos)
A11 = A[0,0]
A12 = A[0,1]
A21 = A[1,0]
A22 = A[1,1]

# Gunakan indeks [row, col] agar dapat angka scalar
B1 = B[0, 0] 
B2 = B[1, 0]

### Desain Gain ###

# Controller Gain (K) - Full State Feedback
poles_controller = [-6, -7] 
res_k = place_poles(A, B, poles_controller)
K = res_k.gain_matrix

# Reduced Observer Gain (L)
# Kita hanya butuh 1 pole karena hanya mengestimasi 1 state (kecepatan)
pole_reduced = [-20] 

# Rumus L untuk Reduced Order (s - (A22 - L*A12) = 0)
L_scalar = (A22 - pole_reduced[0]) / A12

print(f"Controller K: {K}")
print(f"Reduced Observer L: {L_scalar}")

### Implementasi Kendali ###

# Variabel Global untuk Reduced Observer (State 'z')
z_state = 0.0 

# Placeholder untuk sistem kendali yang didefinisikan pengguna
def control_system(t, y_measurement):
    global z_state
    
    # Recover kecepatan (v_hat) 
    # Rumus: v_hat = z + L * y
    v_hat = z_state + L_scalar * y_measurement
    
    # Susun state lengkap [posisi_asli, kecepatan_estimasi]
    # Gunakan posisi asli karena sensor dianggap akurat (Reduced Order)
    full_state_estimate = np.array([y_measurement, v_hat])
    
    # 3. Hitung Force (u = -K * state)
    u = -np.dot(K, full_state_estimate)
    
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
    F = control_system(t, x)
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

# Inisialisasi z agar v_hat awal = 0
# 0 = z + L*y0  ->  z = -L*y0
z_state = -L_scalar * initial_state[0]

solution_time = [0]
solution_states = [current_state]
solution_estimated = [current_state.copy()]

# Fungsi solver untuk memperbarui state
def update_solution():
    global current_state, solution_time, solution_states, z_state
    
    t_curr = solution_time[-1]
    
    # --- Update Observer ---
    # Ambil data yang dibutuhkan
    y_measurement = current_state[0]                # Posisi asli (y)
    u_val = control_system(t_curr, y_measurement)   # Gaya yang diberikan (u)
    
   # Rumus Observer : dz/dt = (A22 - L*A12)*z + (Gain_Y)*y + (Gain_U)*u
    
    # Hitung gain matriks sementara
    AR = A22 - L_scalar * A12
    AY = AR * L_scalar + A21 - L_scalar * A11
    BU = B2 - L_scalar * B1
    
    dz = AR * z_state + AY * y_measurement + BU * u_val
    
    # Update Z
    z_state = z_state + dz * dt
    
    # Update Plant (Mobil)
    sol = solve_ivp(mountain_car_dynamics, [t_curr, t_curr + dt], current_state, t_eval=[t_curr + dt])
    current_state = sol.y[:, -1]
    
    # Simpan Data
    solution_time.append(sol.t[-1])
    solution_states.append(current_state)
    
    # Rekonstruksi v_hat untuk grafik
    v_hat_now = z_state + L_scalar * current_state[0]
    solution_estimated.append([current_state[0], v_hat_now])

# Parameter simulasi
t_span = (0, 15)  # Simulasikan selama 15 detik
dt = 0.01         # Langkah waktu

### VISUALISASI ###
fig = plt.figure(figsize=(10, 8))
fig.suptitle("Challenge 2: Reduced-Order Compensator")

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
ax_pos.set_title("Position Response")
ax_pos.grid(True)

# Subplot 3: Kecepatan
ax_vel = fig.add_subplot(2, 2, 4)
ax_vel.set_xlim(0, 15)
ax_vel.set_ylim(-3, 3)
ax_vel.set_xlabel("Time (s)")
ax_vel.set_ylabel("Velocity (m/s)")
ax_vel.set_title("Estimator Performance (Velocity)")
ax_vel.grid(True)

# Aset Gambar
mountain_x = np.linspace(-1.2, 1.2, 500)
mountain_y = np.sin(k * (mountain_x + peak_shift)) / k
ax_anim.plot(mountain_x, mountain_y, 'k-', lw=2)
car, = ax_anim.plot([], [], 'ro', markersize=10, label='Real Car')
est_car, = ax_anim.plot([], [], 'bx', markersize=8, alpha=0.5, label='Reduced Est')
ax_anim.legend(loc='upper right')

line_pos, = ax_pos.plot([], [], 'b-', lw=2, label='True Position')
ax_pos.plot([0, 15], [0, 0], 'r--', alpha=0.7, label='Target')
ax_pos.legend()

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