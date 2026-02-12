import numpy as np

# ============================================================
# Studi Kasus Sederhana - Linear Regression (Gradient Descent)
# ============================================================

# --- Data ---
x = np.array([2, 5, 4, 7, 8])
y = np.array([8, 17, 13, 24, 26])
m = len(x)  # jumlah data

# --- Inisialisasi Parameter ---
w0 = 0.0  # bias
w1 = 0.0  # weight
alpha = 0.01  # learning rate
epochs = 1000  # jumlah iterasi

# --- Hipotesis: h_w(x) = w0 + w1 * x ---
def hypothesis(w0, w1, x):
    return w0 + w1 * x

# --- Cost Function: J(w0, w1) = (1/2m) * Σ(h_w(x_i) - y_i)^2 ---
def cost_function(w0, w1, x, y, m):
    predictions = hypothesis(w0, w1, x)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# --- Gradient Descent ---
print("=" * 60)
print("GRADIENT DESCENT - LINEAR REGRESSION")
print("=" * 60)
print(f"\nData:")
print(f"  x = {x}")
print(f"  y = {y}")
print(f"\nHyperparameter:")
print(f"  Learning Rate (alpha) = {alpha}")
print(f"  Epochs             = {epochs}")
print(f"\nParameter Awal:")
print(f"  w0 = {w0}, w1 = {w1}")
print(f"  Cost Awal = {cost_function(w0, w1, x, y, m):.6f}")
print("-" * 60)

for epoch in range(epochs):
    # Hitung prediksi
    predictions = hypothesis(w0, w1, x)

    # Hitung error
    error = predictions - y

    # Update Rule:
    # w0 = w0 - α * (1/m) * Σ(h_w(x_i) - y_i) * x0_i   (x0 = 1)
    # w1 = w1 - α * (1/m) * Σ(h_w(x_i) - y_i) * x1_i
    gradient_w0 = (1 / m) * np.sum(error * 1)   # x0^(i) = 1
    gradient_w1 = (1 / m) * np.sum(error * x)    # x1^(i) = x

    w0 = w0 - alpha * gradient_w0
    w1 = w1 - alpha * gradient_w1

    # Cetak progress setiap 100 epoch
    if (epoch + 1) % 100 == 0 or epoch == 0:
        cost = cost_function(w0, w1, x, y, m)
        print(f"  Epoch {epoch+1:>4d} | w0 = {w0:.6f} | w1 = {w1:.6f} | Cost = {cost:.6f}")

# --- Hasil Akhir ---
print("-" * 60)
print(f"\nHasil Akhir setelah {epochs} epoch:")
print(f"  w0 (bias)   = {w0:.6f}")
print(f"  w1 (weight) = {w1:.6f}")
print(f"  Cost Akhir  = {cost_function(w0, w1, x, y, m):.6f}")
print(f"\n  Model: h(x) = {w0:.4f} + {w1:.4f} * x")

# --- Prediksi ---
print("\n" + "=" * 60)
print("PREDIKSI vs AKTUAL")
print("=" * 60)
print(f"  {'x':>3s} | {'y aktual':>10s} | {'y prediksi':>12s} | {'error':>8s}")
print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")
for i in range(m):
    y_pred = hypothesis(w0, w1, x[i])
    err = y_pred - y[i]
    print(f"  {x[i]:>3d} | {y[i]:>10d} | {y_pred:>12.4f} | {err:>8.4f}")

# --- Prediksi data baru ---
print("\n" + "=" * 60)
print("PREDIKSI DATA BARU")
print("=" * 60)
x_new = [3, 6, 10]
for xn in x_new:
    y_pred = hypothesis(w0, w1, xn)
    print(f"  x = {xn:>2d}  =>  h(x) = {y_pred:.4f}")

print()
