import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt

# Параметры задачи (здесь все взято аналогично прошлой задаче)
T = 5.0
th_start = np.array([0.5, 0.5])
th_end = np.array([1.5, -1.0])
jIdx = [1, 3]
dt = 1/240

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -10)
robotId = p.loadURDF("./two_link.urdf.xml", useFixedBase=True)

# Отключаем моторы для torque control
for i in jIdx:
    p.setJointMotorControl2(robotId, i, p.VELOCITY_CONTROL, force=0)

history = {
    't': [],
    'theta_real': [], 'theta_ref': [],
    'vel_real': [], 'vel_ref': []
}
"""Словарь для логирования: t - момент времени
                            theta_real - текущие значения параметров theta
                            theta_ref - следющие (желаемые для управления на каждом шаге) значения параметров theta
                            vel_real - текущие обобщенные скорости
                            vel_ref - следющие (желаемые для управления на каждом шаге) обобщенные скорости"""

# Начальное положение
for i, val in enumerate(th_start):
    p.resetJointState(robotId, jIdx[i], val)

def trapezoidal_time_scaling(t, T, v=1.5):
    """
    Трапецеидальный профиль скорости (Trapezoidal Motion Profile).
    v: максимальная скорость (должна быть > 1/T и <= 2/T для выполнения условий)
    """
    if t < 0: return 0, 0, 0
    if t > T: return 1, 0, 0

    # Согласно формулам из книги, для симметиричности профиля требуется:
    # Ускорение a = v**2 / (v*T - 1)
    # Время ускорения ta = (v*T - 1) / v
    a = (v**2) / (v*T - 1)
    ta = (v*T - 1) / v
    
    if t <= ta:
        # Фаза ускорения
        s = 0.5 * a * t**2
        ds = a * t
        dds = a
    elif t <= T - ta:
        # Фаза постоянной скорости
        s = a * ta**2 / 2 + v * (t - ta)
        ds = v
        dds = 0
    else:
        # Фаза замедления
        tr = T - t # оставшееся время
        s = 1 - 0.5 * a * tr**2
        ds = a * tr
        dds = -a
        
    return s, ds, dds

# Аналогично, управление будет реализовано через ПД-регулятор
Kp = 150.0
Kd = 25.0

# Цикл управления
for step in range(int(T / dt) + 100):
    curr_t = step * dt
    
    # 1. Получение текущего состояния
    states = p.getJointStates(robotId, jIdx)
    theta = np.array([states[0][0], states[1][0]])
    theta_dot = np.array([states[0][1], states[1][1]])
    
    # 2. Траектория: Trapezoidal Profile
    # Выбираем v так, чтобы профиль был реализуем (1/T < v <= 2/T)
    # При T=5, v должно быть > 0.2. Возьмем v = 0.25 для плавности.
    s, ds, dds = trapezoidal_time_scaling(curr_t, T, v=0.25)
    
    # 3. Расчет значений параметров (обобщенных координат)
    theta_d = th_start + s * (th_end - th_start)
    theta_dot_d = ds * (th_end - th_start)
    theta_ddot_d = dds * (th_end - th_start)
    
    # 3. Непосредственно управление по форме реализовано так же, как в прошлой задаче (11.4.2 из книги)
    u = theta_ddot_d + Kp * (theta_d - theta) + Kd * (theta_dot_d - theta_dot)
    
    M = np.array(p.calculateMassMatrix(robotId, list(theta)))
    h = np.array(p.calculateInverseDynamics(robotId, list(theta), list(theta_dot), [0, 0]))
    
    tau = M @ u + h
    
    p.setJointMotorControlArray(robotId, jIdx, p.TORQUE_CONTROL, forces=tau)

    # Логирование
    history['t'].append(curr_t)
    history['theta_real'].append(theta)
    history['theta_ref'].append(theta_d)
    history['vel_real'].append(theta_dot)
    history['vel_ref'].append(theta_dot_d)

    p.stepSimulation()
    time.sleep(dt)

p.disconnect()

history['theta_real'] = np.array(history['theta_real'])
history['theta_ref'] = np.array(history['theta_ref'])
history['vel_real'] = np.array(history['vel_real'])
history['vel_ref'] = np.array(history['vel_ref'])

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Joint Trajectories (Trapezoidal Profile Scaling)', fontsize=16)

for i in range(2):
    # Позиция
    axs[0, i].plot(history['t'], history['theta_ref'][:, i], 'r--', label='Reference')
    axs[0, i].plot(history['t'], history['theta_real'][:, i], 'b', label='Actual', alpha=0.7)
    axs[0, i].set_title(f'Joint {i+1} Position [rad]')
    axs[0, i].legend()
    axs[0, i].grid(True)

    # Скорость
    axs[1, i].plot(history['t'], history['vel_ref'][:, i], 'r--', label='Reference')
    axs[1, i].plot(history['t'], history['vel_real'][:, i], 'g', label='Actual', alpha=0.7)
    axs[1, i].set_title(f'Joint {i+1} Velocity [rad/s]')
    axs[1, i].legend()
    axs[1, i].grid(True)

plt.tight_layout()
plt.show()