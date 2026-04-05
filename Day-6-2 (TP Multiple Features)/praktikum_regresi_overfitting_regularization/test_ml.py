import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

RANDSEED = 42
np.random.seed(RANDSEED)

def add_bias(x):
  bias = np.ones(x.shape[0])
  return np.c_[bias, x]

def cost(y,pred, w=None,lamda=0.0):
    m = y.shape[0]
    mse = ((pred - y) ** 2).sum() / (2 * m)
    reg = 0
    if w is not None and lamda!=0.0:
      reg = (lamda / (2 * m)) * (w[1:] ** 2).sum()   # w0 tidak ikut
    return mse + reg

def predict(w,x):
  return x @ w

def update_bobot(w,xb,y,alpha, lamda=0.0):
  output = predict(w,xb)
  error = output-y
  m = y.shape[0]
  #rergularization
  reg = (lamda / m) * w
  reg[0] = 0
  gradient = (xb.T @ error) / m
  w = w - alpha*(gradient+reg)
  return w

def train(X, y, X_val, y_val, alpha=0.1, iters=500, verbose=True, lamda=0.0):
    np.random.seed(RANDSEED)
    w = np.random.randn(X.shape[1]) * 0.01
    history_train = []
    history_val = []

    for i in range(iters):
        pred = predict(w, X)
        loss = cost(y, pred, w, lamda)
        history_train.append(loss)

        pred_val = predict(w, X_val)
        loss_val = cost(y_val, pred_val)
        history_val.append(loss_val)
        w = update_bobot(w, X, y, alpha, lamda)

    return w, np.array(history_train), np.array(history_val)

def train_test_split(X, y, test_ratio=0.2, seed=RANDSEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def pilih_sample_random(X, y, n, seed=RANDSEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], n, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]
    return X_sample, y_sample

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

#Baca Dataset
data = pd.read_csv("https://drive.google.com/uc?id=16g89ugB-pdYeloUQUhvWLxaIYpTi3e1A")

print("--- PERTANYAAN 1 ---")
print("Jumlah data:", len(data))
print("Nama kolom label:", data.columns[-1] if data.columns[-1] == "harga_juta_rupiah" else "harga_juta_rupiah")
print("Jumlah fitur:", len(data.columns) - 1)
print("Range fitur sama? Check describe:")
# min and max for features
print(data.drop(columns=["harga_juta_rupiah"]).agg(['min', 'max']))

#Pisahkan fitur input dan labelnya
X = data.drop(columns=["harga_juta_rupiah"]).to_numpy()
y = data["harga_juta_rupiah"].to_numpy()

X_min = X.min(axis=0)
X_max = X.max(axis=0)

#Lakukan Scaling
X = (X - X_min) / (X_max - X_min)

X = add_bias(X)

print("\n--- PERTANYAAN 3 ---")
X_train, X_test, y_train, y_test = train_test_split(X,y,0.2)
print("Jml train:", X_train.shape[0])
print("Jml test:", X_test.shape[0])
print("Rasio test:", X_test.shape[0] / X.shape[0])

X_train_sungguhan, X_val, y_train_sungguhan, y_val = train_test_split(X_train,y_train,0.2)

print("\n--- PERTANYAAN 4 ---")
w, hist_train, hist_val = train(X_train_sungguhan, y_train_sungguhan, X_val, y_val, 0.1, 500, False)
hasil_prediksi_train = predict(w,X_train_sungguhan)
error_train = mse(y_train_sungguhan, hasil_prediksi_train)
hasil_prediksi_test = predict(w,X_test)
error_test = mse(y_test, hasil_prediksi_test)
print(f"Base - Error Training: {error_train}, Error Testing: {error_test}")

print("Eksperimen Q4:")
for ir, lr in enumerate([0.01, 0.1, 0.5]):
    for i_it, iters in enumerate([500, 1000]):
        w, _, _ = train(X_train_sungguhan, y_train_sungguhan, X_val, y_val, lr, iters, False)
        err_train = mse(y_train_sungguhan, predict(w,X_train_sungguhan))
        err_test = mse(y_test, predict(w,X_test))
        print(f"Iter: {iters}, LR: {lr}, TrainMSE: {err_train}, TestMSE: {err_test}")

print("\n--- PERTANYAAN 5 ---")
idx_list = [
    ("seluruhnya", list(range(13))),
    ("1", [0,1]),
    ("2", [0,2]),
    ("3", [0,3]),
    ("1,2", [0,1,2]),
    ("1,3", [0,1,3]),
    ("2,3", [0,2,3]),
    ("1,2,3", [0,1,2,3])
]
for name, idx in idx_list:
    w, _, _ = train(X_train_sungguhan[:,idx], y_train_sungguhan, X_val[:,idx], y_val, 0.5, 1000, False) # best config from Q4 maybe? let's guess 1000 and 0.5
    err_train = mse(y_train_sungguhan, predict(w,X_train_sungguhan[:,idx]))
    err_test = mse(y_test, predict(w,X_test[:,idx]))
    print(f"Fitur: {name}, Train MSE: {err_train}, Test MSE: {err_test}")

print("\n--- PERTANYAAN 6 ---")
X_train_small, X_val_small, y_train_small, y_val_small = train_test_split(X_train,y_train,0.99)
print("Small data size:", X_train_small.shape[0])

idx = [0,1]
w, _, _ = train(X_train_small[:,idx], y_train_small, X_val_small[:,idx], y_val_small, 0.01, 1000, False)
print(f"1 fitur, iters 1000 Train: {mse(y_train_small, predict(w,X_train_small[:,idx]))}, Test: {mse(y_test, predict(w,X_test[:,idx]))}")

idx = [0,1,2,3,4]
w, _, _ = train(X_train_small[:,idx], y_train_small, X_val_small[:,idx], y_val_small, 0.01, 10000, False)
print(f"5 fitur, iters 10000 Train: {mse(y_train_small, predict(w,X_train_small[:,idx]))}, Test: {mse(y_test, predict(w,X_test[:,idx]))}")

idx = list(range(13))
w, _, _ = train(X_train_small[:,idx], y_train_small, X_val_small[:,idx], y_val_small, 0.01, 10000, False)
print(f"All fitur, iters 10000 Train: {mse(y_train_small, predict(w,X_train_small[:,idx]))}, Test: {mse(y_test, predict(w,X_test[:,idx]))}")

print("\n--- PERTANYAAN 7 SFS ---")
selected = [0]
remaining = list(range(1, 13))
# Just to be sure we do exactly all steps and pick best sequence
best_seq = []
while remaining:
    best_err = float('inf')
    best_f = -1
    for f in remaining:
        curr = selected + [f]
        w, _, _ = train(X_train_small[:,curr], y_train_small, X_val_small[:,curr], y_val_small, 0.01, 10000, False)
        err = mse(y_val_small, predict(w,X_val_small[:,curr]))
        if err < best_err:
            best_err = err; best_f = f
    selected.append(best_f)
    remaining.remove(best_f)
    print(f"Added {best_f}, Val MSE: {best_err}")
print("Final SFS Seq: ", selected)

print("\n--- PERTANYAAN 7 SBFS ---")
selected = list(range(13))
while len(selected) > 1:
    best_err = float('inf')
    worst_f = -1
    for f in selected:
        if f == 0: continue
        curr = [x for x in selected if x != f]
        w, _, _ = train(X_train_small[:,curr], y_train_small, X_val_small[:,curr], y_val_small, 0.01, 10000, False)
        err = mse(y_val_small, predict(w,X_val_small[:,curr]))
        if err < best_err:
            best_err = err; worst_f = f
    selected.remove(worst_f)
    print(f"Removed {worst_f}, Val MSE: {best_err}")
print("Final SBFS Remaining: ", selected)

print("\n--- PERTANYAAN 8 Regularisasi ---")
for lamda in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
    w, _, _ = train(X_train_small, y_train_small, X_val_small, y_val_small, 0.01, 10000, False, lamda)
    err_train = mse(y_train_small, predict(w,X_train_small))
    err_test = mse(y_test, predict(w,X_test))
    print(f"Lamda: {lamda}, Train: {err_train}, Test: {err_test}")

print("\n--- PERTANYAAN 9 Data Tambahan ---")
rasio_jml_data_training = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for rasio in rasio_jml_data_training:
  jml_data = int(rasio*X_train_sungguhan.shape[0])
  if jml_data == 0: continue
  X_train_rasio, y_train_rasio = pilih_sample_random(X_train_sungguhan, y_train_sungguhan, jml_data)
  w, _, _ = train(X_train_rasio, y_train_rasio, X_val, y_val, 0.01, 10000, False)
  err_train = mse(y_train_small, predict(w,X_train_small))
  err_test = mse(y_test, predict(w,X_test))
  print(f"Rasio {rasio} (n={jml_data}): Train: {err_train}, Test:{err_test}")
