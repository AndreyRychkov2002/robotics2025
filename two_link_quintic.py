import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p


# Параметры симуляции и задачи
dt = 1 / 240
T = 5.0  # Заданное время движения
th_start = np.array([0.5, 0.5])  # Начальное положение
th_end = np.array([1.5, -1.0])  # Целевое положение (в joint-space)
jIdx = [1, 3]  # Используем первую и третью оси


physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Отключить лишний GUI
p.setGravity(0, 0, -10)
boxId = p.loadURDF("./two_link.urdf.xml", useFixedBase=True)


history = {
    "t": [],
    "theta_real": [],
    "theta_ref": [],
    "vel_real": [],
    "vel_ref": [],
}
"""
Словарь для логирования:
  - t: момент времени
  - theta_real: текущие значения углов
  - theta_ref: желаемые значения (reference)
  - vel_real: текущие скорости
  - vel_ref: желаемые скорости (reference)
"""


# Отключаем встроенные моторы для прямого управления моментами
for i in jIdx:
    p.setJointMotorControl2(boxId, i, p.VELOCITY_CONTROL, force=0)


# Установка в начальное положение
for i, val in enumerate(th_start):
    p.resetJointState(boxId, jIdx[i], val)


def get_quintic_s(t, T):
    """Quintic time-scaling (equation 9.9).

    Returns (s, ds, dds) — scale parameter and its first two derivatives.
    """

    if t > T:
        t = T
    tau = t / T
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    dds = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / (T ** 2)
    return s, ds, dds


# Управление: PD-регулятор
Kp = 100.0
Kd = 20.0


start_time = time.time()
for step in range(int(T / dt) + 100):  # небольшое время на стабилизацию
    curr_t = step * dt

    # 1) Получение текущего состояния робота
    joint_states = p.getJointStates(boxId, jIdx)
    theta = np.array([joint_states[0][0], joint_states[1][0]])
    theta_dot = np.array([joint_states[0][1], joint_states[1][1]])

    # 2) Расчет желаемой траектории (линейно в joint-space) + временное масштабирование
    s, ds, dds = get_quintic_s(curr_t, T)
    theta_d = th_start + s * (th_end - th_start)
    theta_dot_d = ds * (th_end - th_start)
    theta_ddot_d = dds * (th_end - th_start)

    # 3) Линеаризация обратной связью — вспомогательное управление u (ускорение)
    u = theta_ddot_d + Kp * (theta_d - theta) + Kd * (theta_dot_d - theta_dot)

    # Матрица инерции M и вектор гравитационно/корректирующих усилий h
    M = np.array(p.calculateMassMatrix(boxId, list(theta)))
    h = np.array(p.calculateInverseDynamics(boxId, list(theta), list(theta_dot), [0, 0]))

    # Итоговый момент по формуле (11.4.2)
    tau = M @ u + h

    # 4) Применение моментов к joint'ам
    p.setJointMotorControlArray(
        bodyIndex=boxId, jointIndices=jIdx, controlMode=p.TORQUE_CONTROL, forces=tau
    )

    # Логирование
    history["t"].append(curr_t)
    history["theta_real"].append(theta)
    history["theta_ref"].append(theta_d)
    history["vel_real"].append(theta_dot)
    history["vel_ref"].append(theta_dot_d)

    p.stepSimulation()
    time.sleep(dt)


p.disconnect()

history["theta_real"] = np.array(history["theta_real"])
history["theta_ref"] = np.array(history["theta_ref"])
history["vel_real"] = np.array(history["vel_real"])
history["vel_ref"] = np.array(history["vel_ref"])


fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Joint Trajectories (Quintic Polynomial Scaling)", fontsize=16)

for i in range(2):
    # Позиция
    axs[0, i].plot(history["t"], history["theta_ref"][:, i], "r--", label="Reference")
    axs[0, i].plot(history["t"], history["theta_real"][:, i], "b", label="Actual", alpha=0.7)
    axs[0, i].set_title(f"Joint {i + 1} Position [rad]")
    axs[0, i].legend()
    axs[0, i].grid(True)

    # Скорость
    axs[1, i].plot(history["t"], history["vel_ref"][:, i], "r--", label="Reference")
    axs[1, i].plot(history["t"], history["vel_real"][:, i], "g", label="Actual", alpha=0.7)
    axs[1, i].set_title(f"Joint {i + 1} Velocity [rad/s]")
    axs[1, i].legend()
    axs[1, i].grid(True)

plt.tight_layout()
plt.show()