# 📚 Strategi Belajar UTS Machine Learning

> [!IMPORTANT]
> UTS ini adalah **praktikum offline** (tes coding tanpa internet). Kamu harus menghafal **rumus**, **pola kode**, dan **konsep** karena tidak bisa browsing saat ujian.

---

## 🗂️ Peta Materi yang Diujikan

Berdasarkan materi kuliah dan format soal praktikum sebelumnya, UTS kemungkinan besar mencakup:

| # | Topik                                | Sumber Materi              | Level Pemahaman |
| - | ------------------------------------ | -------------------------- | --------------- |
| 1 | **Logistic Regression**        | Day-4, Day-4-2 (TP Logreg) | ⭐⭐⭐ Kritis   |
| 2 | **Multiple Features**          | Day-5, Day-6               | ⭐⭐⭐ Kritis   |
| 3 | **Overfitting & Underfitting** | Day-5, Day-6               | ⭐⭐⭐ Kritis   |
| 4 | **Feature Selection**          | Day-6, Day-6-2 (TP)        | ⭐⭐ Penting    |
| 5 | **Regularization**             | Day-6, Day-6-2 (TP)        | ⭐⭐ Penting    |
| 6 | **Scaling / Normalisasi**      | Day-5, Day-6               | ⭐⭐ Penting    |
| 7 | **Train/Test/Val Split**       | Day-6-2 (TP)               | ⭐ Dasar        |
| 8 | **MSE, Log-Loss, Evaluasi**    | Day-3, Day-4               | ⭐ Dasar        |

---

## 📅 Jadwal Belajar (Hari Ini sampai Rabu)

### 🔴 Hari 1 (Hari Ini – Minggu): TEORI & RUMUS

**Tujuan:** Kuasai semua rumus dan konsep inti

#### Sesi 1: Logistic Regression (2 jam)

1. Pahami **sigmoid function**: `σ(z) = 1 / (1 + e^(-z))`
2. Pahami **hipotesis logistic regression**: `h(x) = σ(w · x)`
3. Hafal **Log-Loss (Binary Cross-Entropy)**:
   ```
   J(w) = -1/m * Σ [ y·log(h(x)) + (1-y)·log(1-h(x)) ]
   ```
4. Hafal **Gradient untuk update bobot**:
   ```
   ∂J/∂wⱼ = 1/m * Σ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾
   ```
5. Pahami **threshold** dan efeknya terhadap precision/recall

#### Sesi 2: Multiple Features & Scaling (2 jam)

1. **Kenapa perlu scaling?**
   - Fitur dengan range besar mendominasi gradient
   - Konvergensi jadi lambat tanpa scaling
2. **Min-Max Scaling**: `x' = (x - min) / (max - min)`
3. **Efek jumlah fitur**:
   - Lebih banyak fitur → model lebih kompleks → error lebih rendah (umumnya)
   - Tapi: terlalu banyak fitur + sedikit data = **overfitting**

#### Sesi 3: Overfitting & Underfitting (2 jam)

1. **Underfitting** (High Bias):

   - Train loss TINGGI, Val loss TINGGI
   - Gap antara keduanya KECIL
   - Model terlalu sederhana (fitur sedikit, iterasi kurang)
2. **Overfitting** (High Variance):

   - Train loss RENDAH, Val loss TINGGI
   - Gap antara keduanya BESAR (dan membesar seiring iterasi)
   - Model terlalu kompleks untuk jumlah data yang ada
3. **Good Fit**:

   - Train loss dan Val loss keduanya rendah dan berdekatan

> [!TIP]
> **Cara mudah mengingat:**
>
> - Underfitting = "Model KURANG PINTAR" → error tinggi di mana-mana
> - Overfitting = "Model HAFAL soal tapi GAGAL ujian" → error rendah di training, tinggi di test

---

### 🟡 Hari 2 (Besok – Senin): CODING PRACTICE

**Tujuan:** Hafal pola kode yang sering muncul di soal praktikum

#### Pola Kode #1: Sigmoid Function

```python
import numpy as np
import math

def sigmoid(z):
    return 1 / (1 + math.e ** (-z))
    # Alternatif: return 1 / (1 + np.exp(-z))
```

#### Pola Kode #2: Log-Loss

```python
def log_loss(y, pred):
    m = len(y)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)  # hindari log(0)
    loss = -1/m * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return loss
```

#### Pola Kode #3: Cost Function (MSE) dengan Regularization

```python
def cost(y, pred, w=None, lamda=0.0):
    m = y.shape[0]
    mse = ((pred - y) ** 2).sum() / (2 * m)
    reg = 0
    if w is not None and lamda != 0.0:
        reg = (lamda / (2 * m)) * (w[1:] ** 2).sum()  # w0 tidak ikut
    return mse + reg
```

#### Pola Kode #4: Predict (Linear Regression)

```python
def predict(w, x):
    return x @ w
```

#### Pola Kode #5: Predict (Logistic Regression)

```python
def predict(w, x):
    return sigmoid(x @ w)
```

#### Pola Kode #6: Update Bobot (Gradient Descent) dengan Regularization

```python
def update_bobot(w, xb, y, alpha, lamda=0.0):
    output = predict(w, xb)
    error = output - y
    m = y.shape[0]
  
    # Regularization term
    reg = (lamda / m) * w
    reg[0] = 0  # bias tidak diregularisasi
  
    gradient = (xb.T @ error) / m
    w = w - alpha * (gradient + reg)
    return w
```

#### Pola Kode #7: Add Bias Column

```python
def add_bias(x):
    bias = np.ones(x.shape[0])
    return np.c_[bias, x]
```

#### Pola Kode #8: Min-Max Scaling

```python
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)
```

#### Pola Kode #9: Train/Test Split Manual

```python
def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

#### Pola Kode #10: Training Loop

```python
def train(X, y, X_val, y_val, alpha=0.1, iters=500, lamda=0.0):
    np.random.seed(42)
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
```

#### Pola Kode #11: Evaluasi (Classification)

```python
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def predict_label(X, w, t=0.5):
    return (predict(w, X) >= t).astype(int)

y_pred = predict_label(X, w, t=0.5)
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
```

#### Pola Kode #12: Feature Selection

```python
# Pilih fitur tertentu (ingat: kolom 0 selalu bias)
idx_fitur = [0, 1, 2, 3, 4]  # bias + fitur 1-4
X_selected = X_train[:, idx_fitur]
```

---

### 🟢 Hari 3 (Selasa): LATIHAN SOAL SIMULASI

**Tujuan:** Simulasi UTS dengan soal di bawah ini

---

## 📝 LATIHAN SOAL UTS (Simulasi)

### Soal 1: Implementasi Sigmoid [10 poin]

Implementasikan fungsi sigmoid:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

```python
def sigmoid(z):
    # TODO: implementasikan sigmoid
    pass
```

**Test:**

```python
assert abs(sigmoid(0) - 0.5) < 1e-6
assert sigmoid(10) > 0.99
assert sigmoid(-10) < 0.01
```

> [!NOTE]
> **Jawaban:** `return 1 / (1 + math.e ** (-z))`

---

### Soal 2: Implementasi Log-Loss [15 poin]

Implementasikan fungsi log-loss:

$$
J(w) = -\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_w(x^{(i)}) + (1-y^{(i)}) \log(1 - h_w(x^{(i)}))\right]
$$

```python
def log_loss(y, pred):
    # TODO: implementasikan log-loss
    pass
```

> [!NOTE]
> **Jawaban:**
>
> ```python
> def log_loss(y, pred):
>     m = len(y)
>     pred = np.clip(pred, 1e-15, 1 - 1e-15)
>     loss = -1/m * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
>     return loss
> ```

---

### Soal 3: Implementasi Cost MSE dengan Regularization [15 poin]

Implementasikan fungsi cost MSE yang mendukung regularization L2:

$$
J(w) = \frac{1}{2m}\sum(h(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2
$$

**Catatan:** `w₀` (bias) TIDAK diregularisasi.

```python
def cost(y, pred, w=None, lamda=0.0):
    # TODO: implementasikan
    pass
```

> [!NOTE]
> **Jawaban:**
>
> ```python
> def cost(y, pred, w=None, lamda=0.0):
>     m = y.shape[0]
>     mse = ((pred - y) ** 2).sum() / (2 * m)
>     reg = 0
>     if w is not None and lamda != 0.0:
>         reg = (lamda / (2 * m)) * (w[1:] ** 2).sum()
>     return mse + reg
> ```

---

### Soal 4: Update Bobot dengan Regularization [15 poin]

Implementasikan gradient descent update rule yang mendukung regularization:

```python
def update_bobot(w, xb, y, alpha, lamda=0.0):
    # TODO: implementasikan
    pass
```

> [!NOTE]
> **Jawaban:**
>
> ```python
> def update_bobot(w, xb, y, alpha, lamda=0.0):
>     output = predict(w, xb)
>     error = output - y
>     m = y.shape[0]
>     reg = (lamda / m) * w
>     reg[0] = 0
>     gradient = (xb.T @ error) / m
>     w = w - alpha * (gradient + reg)
>     return w
> ```

---

### Soal 5: Pertanyaan Teori [20 poin]

**a)** Jelaskan perbedaan antara **underfitting** dan **overfitting**. Bagaimana ciri-cirinya pada plot training curve?

> [!NOTE]
> **Jawaban:**
>
> - **Underfitting:** Train loss dan Val loss sama-sama tinggi, gap kecil. Model terlalu sederhana.
> - **Overfitting:** Train loss rendah tapi Val loss tinggi, gap besar dan membesar. Model terlalu kompleks untuk data yang ada.

**b)** Sebutkan 3 cara untuk memperbaiki overfitting!

> [!NOTE]
> **Jawaban:**
>
> 1. **Menambah jumlah data training** → Model punya lebih banyak pola untuk dipelajari
> 2. **Mengurangi jumlah fitur (Feature Selection)** → Mengurangi kompleksitas model
> 3. **Regularization (L2/Ridge)** → Menambahkan penalty pada bobot besar sehingga model tidak terlalu fit pada noise

**c)** Kenapa kita perlu melakukan scaling pada fitur sebelum training?

> [!NOTE]
> **Jawaban:** Agar semua fitur memiliki rentang nilai yang seragam, sehingga:
>
> - Gradient descent konvergen lebih cepat
> - Fitur dengan range besar tidak mendominasi perhitungan gradient
> - Model mendapatkan bobot yang lebih optimal

**d)** Apa efek menaikkan threshold dari 0.5 menjadi 0.7 pada logistic regression?

> [!NOTE]
> **Jawaban:** Model jadi lebih "ketat" dalam memprediksi kelas positif:
>
> - Precision meningkat (lebih sedikit false positive)
> - Recall menurun (lebih banyak false negative)
> - Berguna ketika cost false positive lebih tinggi dari false negative

**e)** Apa perbedaan antara data training, validation, dan testing?

> [!NOTE]
> **Jawaban:**
>
> - **Training**: Digunakan untuk melatih model (update bobot)
> - **Validation**: Digunakan selama training untuk monitor overfitting (tidak update bobot)
> - **Testing**: Digunakan setelah training selesai untuk evaluasi final kemampuan generalisasi

---

### Soal 6: Eksperimen Feature Selection [25 poin]

Diberikan dataset dengan 12 fitur. Lakukan eksperimen dengan kombinasi fitur berikut dan catat hasilnya:

| Eksperimen | Fitur yang digunakan |
| ---------- | -------------------- |
| 1          | Fitur 1 saja         |
| 2          | Fitur 1 dan 2        |
| 3          | Fitur 1, 2, 3        |
| 4          | Fitur 1, 2, 3, 4     |
| 5          | Semua fitur (1-12)   |

**Pertanyaan:**

1. Mana yang memberikan MSE terbaik?
2. Apakah selalu lebih baik menggunakan lebih banyak fitur? Jelaskan!
3. Apa yang terjadi jika jumlah fitur terlalu banyak relatif terhadap jumlah data?

> [!NOTE]
> **Jawaban:**
>
> 1. Biasanya menggunakan fitur lebih banyak memberikan MSE lebih baik, DENGAN CATATAN jumlah data cukup banyak.
> 2. TIDAK selalu. Jika data sedikit dan fitur banyak → **overfitting**
> 3. Model akan **overfitting** — menghafal data training (error rendah) tapi gagal pada data baru (error tinggi)

---

## 🎯 Checklist Persiapan Akhir

- [ ] Hafal rumus sigmoid
- [ ] Hafal rumus log-loss
- [ ] Hafal rumus MSE + regularization
- [ ] Hafal rumus gradient descent + regularization
- [ ] Pahami pola kode add_bias
- [ ] Pahami pola kode scaling (min-max)
- [ ] Pahami pola kode train/test split
- [ ] Pahami pola kode training loop
- [ ] Pahami pola kode evaluasi (confusion matrix, accuracy, precision, recall, F1)
- [ ] Pahami konsep underfitting vs overfitting
- [ ] Pahami 3 solusi overfitting (tambah data, feature selection, regularization)
- [ ] Pahami kenapa perlu scaling
- [ ] Pahami efek threshold pada logistic regression
- [ ] Pahami perbedaan train/val/test set

> [!CAUTION]
> **Jangan lupa:**
>
> - `np.clip(pred, 1e-15, 1-1e-15)` pada log-loss untuk menghindari `log(0)`
> - Bias (`w[0]`) **TIDAK** diregularisasi
> - Kolom bias (kolom semua 1) harus ditambahkan ke X sebelum training
> - Scaling dilakukan SEBELUM menambah kolom bias

Good luck untuk UTS-nya! 🍀
