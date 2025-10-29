import numpy as np
import matplotlib.pyplot as plt

# 1. Ввод параметров
print("Настройка данных")
n_points = int(input("Введите количество точек для каждого класса (например, 15): ") or 15)

x1_c1 = float(input("Введите X центра 1-го класса (по умолчанию 5): ") or 5)
x2_c1 = float(input("Введите Y центра 1-го класса (по умолчанию -5): ") or -5)
x1_c2 = float(input("Введите X центра 2-го класса (по умолчанию -5): ") or -5)
x2_c2 = float(input("Введите Y центра 2-го класса (по умолчанию 5): ") or 5)

distribution = input("Тип распределения (normal/uniform)? [normal]: ") or "normal"
spread = float(input("Введите стандартное отклонение / диапазон (по умолчанию 1.0): ") or 1.0)

np.random.seed(42)

# 2. Генерация точек
center_1 = np.array([x1_c1, x2_c1])
center_2 = np.array([x1_c2, x2_c2])

if distribution == "uniform":
    class1 = center_1 + np.random.uniform(-spread, spread, size=(n_points, 2))
    class2 = center_2 + np.random.uniform(-spread, spread, size=(n_points, 2))
else:
    class1 = center_1 + np.random.normal(0.0, spread, size=(n_points, 2))
    class2 = center_2 + np.random.normal(0.0, spread, size=(n_points, 2))

X_raw = np.vstack([class1, class2])
bias = np.ones((2 * n_points, 1))
X = np.hstack([X_raw, bias])
y = np.hstack([np.ones(n_points), np.zeros(n_points)])

# 3. График точек (по запросу)
if input("Показать сгенерированные точки? (y/n): ").lower() == "y":
    plt.scatter(class1[:, 0], class1[:, 1], marker='o', label='Class 1 (y=1)')
    plt.scatter(class2[:, 0], class2[:, 1], marker='s', label='Class 0 (y=0)')
    plt.title("Сгенерированные точки двух классов")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

# 4. Определяем функции
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_raw(X, w):
    return X @ w

def loss_mse_smooth(X, y, w):
    z = predict_raw(X, w)
    y_tilde = sigmoid(z)
    return np.sum((y_tilde - y) ** 2)

def grad_mse_smooth(X, y, w):
    z = predict_raw(X, w)
    y_tilde = sigmoid(z)
    diff = 2 * (y_tilde - y) * y_tilde * (1 - y_tilde)
    return X.T @ diff

# 5. Обучение
print("\n Обучение модели")
w = np.array([0.5, 1.5, 0.0])
lr = float(input("Введите скорость обучения (по умолчанию 0.05): ") or 0.05)
epochs = int(input("Введите число эпох (по умолчанию 2000): ") or 2000)

loss_history = []
for _ in range(epochs):
    g = grad_mse_smooth(X, y, w)
    w -= lr * g
    loss_history.append(loss_mse_smooth(X, y, w))

print(f"Обученные веса: {w}")
print(f"Финальная невязка: {loss_history[-1]:.6f}")

# 6. График разделяющей прямой (по запросу)
def plot_decision_boundary(w, class1, class2):
    plt.scatter(class1[:, 0], class1[:, 1], marker='o', label='Class 1 (y=1)')
    plt.scatter(class2[:, 0], class2[:, 1], marker='s', label='Class 0 (y=0)')
    x1_vals = np.linspace(-10, 10, 100)
    w1, w2, b = w
    if abs(w2) < 1e-8:
        plt.axvline(-b / w1, color='r')
    else:
        x2_vals = -(w1 / w2) * x1_vals - b / w2
        plt.plot(x1_vals, x2_vals, 'r-')
    plt.title("Разделяющая прямая после обучения")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

if input("Показать разделяющую прямую? (y/n): ").lower() == "y":
    plot_decision_boundary(w, class1, class2)

# 7. График невязки (по запросу)
if input("Показать график невязки? (y/n): ").lower() == "y":
    plt.plot(loss_history)
    plt.title("Сходимость невязки")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка (MSE)")
    plt.grid(True)
    plt.show()

print("\nПрограмма завершена.")
