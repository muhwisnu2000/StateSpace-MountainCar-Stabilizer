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
    """
    Fungsi ini menggabungkan Observer dan Controller.
    Input: y_measurement (Hanya posisi x dari sensor)
    Output: Force (F)
    """
    
    global observer_state, last_t
    
    # Hitung dt (delta time) untuk update integral observer
    dt = t - last_t
    if dt <= 0: dt = 0.01 # Fallback untuk langkah pertama
    last_t = t

    # 1. STATE ESTIMATOR (OBSERVER)
    # Rumus: d(x_hat)/dt = A*x_hat + B*u + L(y - C*x_hat)
    # y = y_measurement (posisi aktual dari sensor)
    
    # Prediksi Output Estimasi (y_hat)
    y_hat = np.dot(C, observer_state) # C * x_hat
    
    # Error Estimasi (Sensor - Estimasi)
    estimation_error = y_measurement - y_hat
    
    # Hitung Kontrol (u) berdasarkan state estimasi SEBELUM update
    # u = -K * x_hat
    u = -np.dot(K, observer_state)
    
    # Hitung Turunan State Estimasi (d_x_hat)
    # A*x_hat
    term1 = np.dot(A, observer_state)
    # B*u
    term2 = np.dot(B, u).flatten() # flatten agar jadi vektor 1D
    # L(y - y_hat)
    term3 = np.dot(L, estimation_error).flatten()
    
    d_x_hat = term1 + term2 + term3
    
    # Update State Estimasi (Integrasi Euler Sederhana)
    observer_state = observer_state + d_x_hat * dt
    
    # Kembalikan gaya u (scalar)
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
initial_state = [-0.5, 0.0] # Mulai acak
current_state = initial_state.copy()
observer_state = np.array([0.0, 0.0]) # Reset observer
last_t = 0
solution_time = [0]
solution_states = [current_state]
solution_estimated = [observer_state.copy()]

# Fungsi solver untuk memperbarui state
def update_solution():
    global current_state, solution_time, solution_states, last_t
    t_curr = solution_time[-1]
    
    # Solve untuk time step berikutnya
    sol = solve_ivp(mountain_car_dynamics, [t_curr, t_curr + dt], current_state, t_eval=[t_curr + dt])
    
    if sol.y.size > 0:
        current_state = sol.y[:, -1]
        solution_time.append(sol.t[-1])
        solution_states.append(current_state)
        solution_estimated.append(observer_state.copy()) # Simpan log estimasi

# Parameter simulasi
t_span = (0, 15)  # Simulasikan selama 15 detik
dt = 0.01         # Langkah waktu

# Visualisasi: Animasikan Mountain Car
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("Position (x)")
ax.set_ylabel("Height")
ax.set_title("Challenge 1: Full-State Compensator (Stabilizing at Peak)")

# Representasi Gunung dan Mobil
mountain_x = np.linspace(-1.2, 1.2, 500)
mountain_y = np.sin(k * (mountain_x + peak_shift)) / k  # Profil gunung yang disesuaikan
car, = ax.plot([], [], 'ro', markersize=8, label='Real Car') # Visualisasi Mobil
est_car, = ax.plot([], [], 'bx', markersize=8, alpha=0.5, label='Estimated State') # Visualisasi Estimator
mountain_line, = ax.plot(mountain_x, mountain_y, 'k-', lw=2)
ax.legend()
ax.grid(True)

# Inisialisasi animasi
def init():
    car.set_data([], [])
    est_car.set_data([], [])
    return car, est_car

# Fungsi update untuk animasi
def update(frame):
    update_solution()
    
    # Real Car
    car_x = current_state[0]
    car_y = np.sin(k * (car_x + peak_shift)) / k
    car.set_data([car_x], [car_y])
    
    # Estimated Car (Melihat apakah estimator akurat)
    est_x = solution_estimated[-1][0]
    est_y = np.sin(k * (est_x + peak_shift)) / k
    est_car.set_data([est_x], [est_y])
    
    return car, est_car

# Buat animasi
ani = animation.FuncAnimation(fig, update, frames=400, init_func=init, blit=True, interval=20)

plt.show()