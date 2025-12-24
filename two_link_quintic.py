import pybullet as p
import numpy as np
import time

# Параметры симуляции и задачи
dt = 1/240
T = 5.0              # Заданное время движения
th_start = np.array([0.5, 0.5])  # Начальное положение
th_end = np.array([1.5, -1.0])   # Целевое положение (в joint-space)
jIdx = [1, 3] # Используем первую и третью оси

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Чтобы отключить отображение графических элементов (оси, сетки, etc)
p.setGravity(0, 0, -10)
boxId = p.loadURDF("./two_link.urdf.xml", useFixedBase=True)

# Отключаем встроенные моторы для прямого управления моментами
for i in jIdx:
    p.setJointMotorControl2(boxId, i, p.VELOCITY_CONTROL, force=0)

# Установка в начальное положение
for i, val in enumerate(th_start):
    p.resetJointState(boxId, jIdx[i], val)

def get_quintic_s(t, T):
    """Полином 5-го порядка (Quintic Polynomial) по формуле (9.9)"""
    if t > T: t = T
    tau = t / T
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5 # функция временного масштабирования
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T # производная функции временного масштабирования
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (T**2) # вторая производная функции временного масштабирования
    return s, ds, dds

# Управление реализовано с помощью ПД-регулятора
Kp = 100.0
Kd = 20.0

start_time = time.time()
for step in range(int(T / dt) + 100): # + небольшое время на стабилизацию
    curr_t = step * dt
    
    # 1. Получение текущего состояния робота
    joint_states = p.getJointStates(boxId, jIdx)
    theta = np.array([joint_states[0][0], joint_states[1][0]])
    theta_dot = np.array([joint_states[0][1], joint_states[1][1]])
    
    # 2. Расчет желаемой траектории как отрезка прямой в joint-space
    s, ds, dds = get_quintic_s(curr_t, T)
    theta_d = th_start + s * (th_end - th_start)
    theta_dot_d = ds * (th_end - th_start)
    theta_ddot_d = dds * (th_end - th_start)
    
    # 3. Линеаризация обратной связью
    # Вспомогательное управление u (ускорение)
    u = theta_ddot_d + Kp * (theta_d - theta) + Kd * (theta_dot_d - theta_dot) # управление ускорением через ПД-регулятор
    
    # Получение матрицы инерции M и вектора сил h с помощью встроенных функций
    M = np.array(p.calculateMassMatrix(boxId, list(theta)))
    # h вычисляется через ID при нулевом ускорении
    h = np.array(p.calculateInverseDynamics(boxId, list(theta), list(theta_dot), [0, 0]))
    
    # Итоговый момент по формуле из 11.4.2
    tau = M @ u + h
    
    # 4. Применение моментов к joint'ам
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jIdx, controlMode=p.TORQUE_CONTROL, forces=tau)
    
    p.stepSimulation()
    time.sleep(dt)

p.disconnect()