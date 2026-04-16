"""
Script to fix and complete the practicum notebook.
This will create a new completed notebook from scratch.
"""
import json

def make_md(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

def make_code(source_lines, execution_count=None, outputs=None):
    cell = {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs if outputs else [],
        "source": source_lines
    }
    return cell

cells = []

# ============================================================
# HEADER
# ============================================================
cells.append(make_md([
    "# MACHINE LEARNING \n",
    "\n",
    "Petunjuk\n",
    "- Jalankan cell dari atas ke bawah.\n",
    "- Isi NIM dan Nama.\n",
    "- Isi bagian TODO.\n"
]))

cells.append(make_code([
    "NAMA = \"Muhammad 'Azmi Salam\"\n",
    "NIM = \"2406010\""
]))

cells.append(make_md(["---"]))

cells.append(make_md([
    "# PRAKTIKUM NON-LINEARITY & ANN\n",
    "\n",
    "Tujuan\n",
    "- Memahami konsep non-linearitas dalam Machine Learning\n",
    "- Mengidentifikasi keterbatasan model linear\n",
    "- Menggunakan Artificial Neural Network (ANN) dan kNN dalam masalah nyata\n",
    "- Memahami peran fungsi aktivasi non-linear dalam ANN\n",
    "- Membandingkan performa model linear vs ANN\n",
    "- Memvisualisasikan decision boundary\n",
    "- Mengeksplorasi pengaruh arsitektur jaringan pada ANN\n",
    "- Mampu menerapkan ANN dalam multiclass classification\n",
    "- Membandingkan ANN vs kNN"
]))

# ============================================================
# IMPORTS
# ============================================================
cells.append(make_code([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.linear_model import LogisticRegression, SGDClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
    "from sklearn.inspection import DecisionBoundaryDisplay\n",
    "\n",
    "RANDSEED = 42\n",
    "listmodel = {}"
]))

# ============================================================
# 1. EKSPERIMEN BINARY CLASSIFICATION
# ============================================================
cells.append(make_md(["## 1. EKSPERIMEN BINARY CLASSIFICATION"]))

# A. PERSIAPAN DATA
cells.append(make_md(["### A. PERSIAPAN DATA"]))

cells.append(make_code([
    "data = pd.read_csv(\"dataset/dataset_kredit_multi.csv\")\n",
    "#Fitur: rasio utang terhadap pendapatan, stabilitas arus kas bulanan\n",
    "#Label: 0 = kredit lancar, 1 = berisiko macet"
]))

cells.append(make_code([
    "#PREVIEW DATA KREDIT\n",
    "data.info()\n",
    "plt.scatter(data[\"rasio_hutang\"], data[\"stabilitas\"], c=data[\"resiko\"])\n",
    "plt.xlabel(\"Rasio\")\n",
    "plt.ylabel(\"Stabilitas\")\n",
    "plt.title(\"Dataset Kredit\")\n",
    "plt.show()"
]))

cells.append(make_code(["data.head()"]))

cells.append(make_code([
    "#Lakukan Normalisasi terhadap Fitur input\n",
    "scaler = MinMaxScaler()\n",
    "data[['rasio_hutang', 'stabilitas']] = scaler.fit_transform(data[['rasio_hutang', 'stabilitas']])\n",
    "data.head()"
]))

cells.append(make_code([
    "#Memisahkan Fitur Input dan Label\n",
    "col = \"resiko\"\n",
    "X = data.drop(columns=[col]).to_numpy()\n",
    "y = data[col].to_numpy()"
]))

cells.append(make_code([
    "#Pembagian dataset menjadi Training dan Testing dengan proporsi 80:20\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDSEED, stratify=y)"
]))

# ============================================================
# B. EKSPERIMEN MODEL LOGREG
# ============================================================
cells.append(make_md(["### B. EKSPERIMEN MODEL LOGREG"]))

# FIX: Changed solver from "liblinear" to "lbfgs" for multiclass support
cells.append(make_code([
    "#Training Model Logreg\n",
    "# Menggunakan solver 'lbfgs' karena 'liblinear' tidak mendukung multiclass (n_classes >= 3)\n",
    "model = LogisticRegression(max_iter=1000, random_state=RANDSEED, C=1, solver=\"lbfgs\")\n",
    "model.nickname=\"LOGREG\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

# FIX: Added average="macro" for multiclass metrics
cells.append(make_code([
    "#Evaluasi Error Traning\n",
    "pred = model.predict(X_train)\n",
    "label = y_train\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# ============================================================
# C. EKSPERIMEN MODEL SGDClassifier
# ============================================================
cells.append(make_md(["### C. EKSPERIMEN MODEL SGDClassifier"]))

cells.append(make_code([
    "#Training Model SGDClassifier\n",
    "model = SGDClassifier(\n",
    "    loss=\"log_loss\",\n",
    "    learning_rate=\"constant\",\n",
    "    eta0=0.01,     # learning rate\n",
    "    alpha=0.01,    # regularisasi\n",
    "    max_iter=100,\n",
    "    random_state=RANDSEED,\n",
    "    verbose=0,\n",
    "    validation_fraction=0.2\n",
    ")\n",
    "model.nickname = \"SGDClassifier\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Traning\n",
    "pred = model.predict(X_train)\n",
    "label = y_train\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# ============================================================
# D. EKSPERIMEN MODEL ANN
# ============================================================
cells.append(make_md(["### D. EKSPERIMEN MODEL ANN"]))

cells.append(make_md([
    "<strong>TODO: </strong> \n",
    "1. Pada setiap eksperimen ubah model nickname supaya tidak mereplace model yang sebelumnya, misal: \"ANN_act_relu\"\n",
    "2. Lakukan eksperimen dengan mengubah-ubah fungsi aktivasi yang digunakan, mana fungsi aktivasi yang terbaik untuk dataset ini? kenapa?\n",
    "3. Lakukan eksperimen dengan mengubah-ubah arsitektur jaringan, mana yang terbaik? kenapa?"
]))

# --- ANN Experiment 1: Identity activation (baseline) ---
cells.append(make_md(["#### Eksperimen 1: ANN dengan aktivasi Identity (baseline)"]))

cells.append(make_code([
    "#Training Model ANN - Identity activation (baseline)\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(4,),\n",
    "    activation=\"identity\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_act_identity\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Traning\n",
    "pred = model.predict(X_train)\n",
    "label = y_train\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# --- ANN Experiment 2: ReLU activation ---
cells.append(make_md(["#### Eksperimen 2: ANN dengan aktivasi ReLU"]))

cells.append(make_code([
    "#Training Model ANN - ReLU activation\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(4,),\n",
    "    activation=\"relu\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_act_relu\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# --- ANN Experiment 3: Tanh activation ---
cells.append(make_md(["#### Eksperimen 3: ANN dengan aktivasi Tanh"]))

cells.append(make_code([
    "#Training Model ANN - Tanh activation\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(4,),\n",
    "    activation=\"tanh\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_act_tanh\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# --- ANN Experiment 4: Logistic activation ---
cells.append(make_md(["#### Eksperimen 4: ANN dengan aktivasi Logistic (Sigmoid)"]))

cells.append(make_code([
    "#Training Model ANN - Logistic activation\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(4,),\n",
    "    activation=\"logistic\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_act_logistic\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# --- ANN Experiment 5: Architecture experiments ---
cells.append(make_md(["#### Eksperimen 5: ANN dengan arsitektur lebih besar - ReLU (8, 8)"]))

cells.append(make_code([
    "#Training Model ANN - Arsitektur lebih besar (8, 8) dengan ReLU\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(8, 8),\n",
    "    activation=\"relu\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_relu_8_8\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# --- ANN Experiment 6: Even deeper ---
cells.append(make_md(["#### Eksperimen 6: ANN dengan arsitektur lebih dalam - ReLU (16, 8, 4)"]))

cells.append(make_code([
    "#Training Model ANN - Arsitektur deep (16, 8, 4) dengan ReLU\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(16, 8, 4),\n",
    "    activation=\"relu\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_relu_16_8_4\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_test\n",
    "yplot = y_test\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# ============================================================
# 2. REKAP EKSPERIMEN
# ============================================================
cells.append(make_md(["## 2. REKAP EKSPERIMEN BINARY CLASSIFICATION"]))

cells.append(make_code([
    "results = []\n",
    "x_data = X_test\n",
    "label = y_test\n",
    "\n",
    "for name, mdl in listmodel.items():\n",
    "    train_pred = mdl.predict(X_train)\n",
    "    test_pred = mdl.predict(X_test)\n",
    "    results.append({\n",
    "        \"model\": name,\n",
    "        \"train_acc\": accuracy_score(y_train, train_pred),\n",
    "        \"train_prec\": precision_score(y_train, train_pred, average=\"macro\"),\n",
    "        \"train_rec\": recall_score(y_train, train_pred, average=\"macro\"),\n",
    "        \"train_f1\": f1_score(y_train, train_pred, average=\"macro\"),\n",
    "        \"test_acc\": accuracy_score(y_test, test_pred),\n",
    "        \"test_prec\": precision_score(y_test, test_pred, average=\"macro\"),\n",
    "        \"test_rec\": recall_score(y_test, test_pred, average=\"macro\"),\n",
    "        \"test_f1\": f1_score(y_test, test_pred, average=\"macro\")\n",
    "    })\n",
    "\n",
    "df = pd.DataFrame(results)\n",
    "df.head(10)"
]))

cells.append(make_md([
    "<strong>TODO: </strong> Jelaskan: \n",
    "1. Mana model yang paling baik jika dilihat dari F1 Score nya? kenapa?\n",
    "2. Jelaskan keterbatasan dan kelebihan masing-masing model?\n",
    "3. Jelaskan karakteristik dari decision boundary yang terbentuk oleh masing-masing model!\n"
]))

cells.append(make_md([
    "**Jawaban:**\n",
    "\n",
    "**1. Model Terbaik berdasarkan F1 Score:**\n",
    "\n",
    "Model ANN dengan fungsi aktivasi non-linear (relu, tanh) dan arsitektur yang lebih dalam (misalnya `ANN_relu_16_8_4` atau `ANN_relu_8_8`) umumnya memberikan F1 Score terbaik pada dataset ini. Hal ini karena dataset kredit memiliki pola non-linear (terlihat dari scatter plot yang menunjukkan distribusi kelas yang tidak dapat dipisahkan oleh garis lurus). Fungsi aktivasi non-linear memungkinkan ANN untuk mempelajari batas keputusan yang kompleks dan melengkung, sehingga mampu memisahkan kelas dengan lebih akurat.\n",
    "\n",
    "**2. Keterbatasan dan Kelebihan masing-masing model:**\n",
    "\n",
    "- **Logistic Regression (LOGREG):**\n",
    "  - Kelebihan: Sederhana, cepat, mudah diinterpretasi, tidak mudah overfitting.\n",
    "  - Keterbatasan: Hanya mampu membuat decision boundary linear, sehingga performanya buruk pada data non-linear.\n",
    "\n",
    "- **SGDClassifier:**\n",
    "  - Kelebihan: Efisien untuk dataset besar, dapat di-optimize secara online (incremental learning).\n",
    "  - Keterbatasan: Sama seperti Logistic Regression, decision boundary tetap linear karena menggunakan loss function log_loss.\n",
    "\n",
    "- **ANN (Identity):**\n",
    "  - Kelebihan: Arsitektur neural network yang fleksibel.\n",
    "  - Keterbatasan: Dengan aktivasi identity, ANN menjadi setara dengan model linear sehingga tidak dapat menangkap pola non-linear.\n",
    "\n",
    "- **ANN (ReLU / Tanh / Logistic):**\n",
    "  - Kelebihan: Mampu mempelajari pola non-linear yang kompleks, decision boundary yang fleksibel.\n",
    "  - Keterbatasan: Memerlukan tuning hyperparameter (arsitektur, learning rate), waktu training lebih lama, risiko overfitting jika arsitektur terlalu besar.\n",
    "\n",
    "**3. Karakteristik Decision Boundary:**\n",
    "\n",
    "- **LOGREG & SGDClassifier:** Membentuk decision boundary berupa **garis lurus** (linear). Boundary ini tidak dapat mengikuti pola distribusi data yang melengkung.\n",
    "- **ANN Identity:** Decision boundary juga **linear** karena fungsi aktivasi identity (f(x) = x) tidak menambahkan non-linearitas.\n",
    "- **ANN ReLU:** Decision boundary berbentuk **potongan garis lurus (piecewise linear)** yang dapat membentuk region keputusan kompleks.\n",
    "- **ANN Tanh/Logistic:** Decision boundary **smooth dan melengkung**, mampu mengikuti pola distribusi data non-linear dengan halus.\n",
    "- Arsitektur yang lebih dalam (lebih banyak layer dan neuron) menghasilkan decision boundary yang **lebih kompleks dan detail**, namun berisiko overfitting."
]))

# ============================================================
# 3. EKSPERIMEN MULTICLASS CLASSIFICATION
# ============================================================
cells.append(make_md(["## 3. EKSPERIMEN MULTICLASS CLASSIFICATION"]))

# A. PERSIAPAN DATA
cells.append(make_md(["### A. PERSIAPAN DATA"]))

# FIX: Fixed the data path (removed leading /)
cells.append(make_code([
    "# Reset listmodel untuk eksperimen multiclass\n",
    "listmodel = {}\n",
    "\n",
    "data = pd.read_csv(\"dataset/dataset_kredit_multi.csv\")\n",
    "#Fitur: rasio utang terhadap pendapatan, stabilitas arus kas bulanan\n",
    "#Label: 0 = kredit lancar, 1 = berisiko macet sedang, 2 = berisiko macet tinggi"
]))

cells.append(make_code([
    "#PREVIEW DATA KREDIT\n",
    "data.info()\n",
    "plt.scatter(data[\"rasio_hutang\"], data[\"stabilitas\"], c=data[\"resiko\"])\n",
    "plt.xlabel(\"Rasio\")\n",
    "plt.ylabel(\"Stabilitas\")\n",
    "plt.title(\"Dataset Kredit\")\n",
    "plt.show()"
]))

cells.append(make_code(["data.head()"]))

cells.append(make_code([
    "#Lakukan Normalisasi terhadap Fitur input\n",
    "scaler = MinMaxScaler()\n",
    "data[['rasio_hutang', 'stabilitas']] = scaler.fit_transform(data[['rasio_hutang', 'stabilitas']])\n",
    "data.head()"
]))

cells.append(make_code([
    "#Memisahkan Fitur Input dan Label\n",
    "col = \"resiko\"\n",
    "X = data.drop(columns=[col]).to_numpy()\n",
    "y = data[col].to_numpy()"
]))

cells.append(make_code([
    "#Pembagian dataset menjadi Training dan Testing dengan proporsi 80:20\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDSEED, stratify=y)"
]))

# B. TRAINING MODEL ANN
cells.append(make_md(["### B. TRAINING MODEL ANN"]))

# ANN Multiclass - Identity (baseline)
cells.append(make_md(["#### ANN Multiclass - Identity (baseline)"]))

cells.append(make_code([
    "#Training Model ANN - Identity (baseline)\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(8,),\n",
    "    activation=\"identity\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_multi_identity\"\n",
    "\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Traning\n",
    "pred = model.predict(X_train)\n",
    "label = y_train\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_train\n",
    "yplot = y_train\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# ANN Multiclass - ReLU
cells.append(make_md(["#### ANN Multiclass - ReLU (8, 8)"]))

cells.append(make_code([
    "#Training Model ANN - ReLU (8, 8) untuk multiclass\n",
    "model = MLPClassifier(\n",
    "    hidden_layer_sizes=(8, 8),\n",
    "    activation=\"relu\",\n",
    "    max_iter=2000,\n",
    "    random_state=RANDSEED\n",
    ")\n",
    "model.nickname = \"ANN_multi_relu_8_8\"\n",
    "\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Traning\n",
    "pred = model.predict(X_train)\n",
    "label = y_train\n",
    "\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_train\n",
    "yplot = y_train\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# C. EVALUASI MODEL
cells.append(make_md(["### C. EVALUASI MODEL "]))

cells.append(make_code([
    "#Evaluasi Error Test - ANN Identity\n",
    "model_eval = listmodel[\"ANN_multi_identity\"]\n",
    "pred = model_eval.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"=== ANN Identity ===\")\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))\n",
    "\n",
    "print()\n",
    "\n",
    "#Evaluasi Error Test - ANN ReLU\n",
    "model_eval = listmodel[\"ANN_multi_relu_8_8\"]\n",
    "pred = model_eval.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"=== ANN ReLU (8,8) ===\")\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk - Test\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "for idx, name in enumerate([\"ANN_multi_identity\", \"ANN_multi_relu_8_8\"]):\n",
    "    mdl = listmodel[name]\n",
    "    plt.sca(axes[idx])\n",
    "    Xplot = X_test\n",
    "    yplot = y_test\n",
    "    DecisionBoundaryDisplay.from_estimator(mdl, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200, ax=axes[idx])\n",
    "    axes[idx].scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "    axes[idx].set_title(name)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(make_md([
    "<strong>TODO: </strong> Jelaskan bagaimana kemampuan ANN dalam melakukan klasifikasi multiclass!"
]))

cells.append(make_md([
    "**Jawaban:**\n",
    "\n",
    "ANN (Artificial Neural Network) memiliki kemampuan yang sangat baik dalam melakukan klasifikasi multiclass, dengan beberapa karakteristik penting:\n",
    "\n",
    "1. **Output Layer Softmax:** Pada klasifikasi multiclass, ANN (MLPClassifier) secara otomatis menggunakan fungsi softmax pada output layer. Softmax mengubah output menjadi probabilitas untuk setiap kelas, sehingga jumlah probabilitas semua kelas = 1. Kelas dengan probabilitas tertinggi dipilih sebagai prediksi.\n",
    "\n",
    "2. **Penanganan Pola Non-Linear:** Dengan fungsi aktivasi non-linear (relu, tanh), ANN mampu membentuk decision boundary yang kompleks dan melengkung. Ini sangat penting pada dataset multiclass di mana batas antar kelas seringkali tidak linear.\n",
    "\n",
    "3. **Perbandingan Identity vs Non-Linear:**\n",
    "   - ANN dengan aktivasi **identity** pada multiclass menghasilkan decision boundary linear yang hanya mampu memisahkan kelas dengan garis lurus. Hasilnya kurang optimal untuk data non-linear.\n",
    "   - ANN dengan aktivasi **relu** atau **tanh** mampu membentuk region keputusan yang lebih fleksibel, sehingga F1 Score meningkat signifikan.\n",
    "\n",
    "4. **Skalabilitas:** ANN secara native mendukung multiclass tanpa perlu strategi tambahan seperti One-vs-Rest (OvR) atau One-vs-One (OvO), karena output layer langsung memiliki neuron sebanyak jumlah kelas.\n",
    "\n",
    "5. **Arsitektur Fleksibel:** Dengan menambah hidden layer dan neuron, ANN dapat mempelajari representasi fitur yang semakin abstrak dan kompleks, meningkatkan kemampuan klasifikasi multiclass."
]))

# D. EKSPERIMEN KNN
cells.append(make_md(["### D. EKSPERIMEN KNN"]))

# KNN n_neighbors=3
cells.append(make_md(["#### KNN dengan n_neighbors = 3"]))

cells.append(make_code([
    "model = KNeighborsClassifier(n_neighbors=3)\n",
    "model.nickname = \"KNN_k3\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model"
]))

cells.append(make_code([
    "#Evaluasi Error Testing\n",
    "pred = model.predict(X_test)\n",
    "label = y_test\n",
    "\n",
    "print(\"=== KNN k=3 ===\")\n",
    "print(\"Accuracy :\", accuracy_score(label, pred))\n",
    "print(\"Precision:\", precision_score(label, pred, average=\"macro\"))\n",
    "print(\"Recall   :\", recall_score(label, pred, average=\"macro\"))\n",
    "print(\"F1 Score :\", f1_score(label, pred, average=\"macro\"))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(label, pred))"
]))

cells.append(make_code([
    "#Lihat Decision Boundary yang terbentuk\n",
    "Xplot = X_train\n",
    "yplot = y_train\n",
    "DecisionBoundaryDisplay.from_estimator(model, Xplot, response_method=\"predict\", cmap='RdBu', alpha=0.3, grid_resolution=200)\n",
    "plt.scatter(Xplot[:, 0], Xplot[:, 1], c=yplot, edgecolors='w')\n",
    "plt.title(model.nickname)\n",
    "plt.show()"
]))

# KNN Experiments with different k
cells.append(make_md(["#### Eksperimen KNN dengan berbagai nilai n_neighbors"]))

cells.append(make_code([
    "# Eksperimen dengan berbagai nilai k\n",
    "k_values = [1, 3, 5, 7, 9, 11, 15, 21]\n",
    "knn_results = []\n",
    "\n",
    "for k in k_values:\n",
    "    knn_model = KNeighborsClassifier(n_neighbors=k)\n",
    "    knn_model.fit(X_train, y_train)\n",
    "    \n",
    "    train_pred = knn_model.predict(X_train)\n",
    "    test_pred = knn_model.predict(X_test)\n",
    "    \n",
    "    train_f1 = f1_score(y_train, train_pred, average=\"macro\")\n",
    "    test_f1 = f1_score(y_test, test_pred, average=\"macro\")\n",
    "    test_acc = accuracy_score(y_test, test_pred)\n",
    "    \n",
    "    knn_results.append({\n",
    "        \"k\": k,\n",
    "        \"train_f1\": train_f1,\n",
    "        \"test_f1\": test_f1,\n",
    "        \"test_acc\": test_acc\n",
    "    })\n",
    "    \n",
    "    print(f\"k={k:2d} | Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}\")\n",
    "\n",
    "# Visualisasi\n",
    "knn_df = pd.DataFrame(knn_results)\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.plot(knn_df['k'], knn_df['train_f1'], 'o-', label='Train F1')\n",
    "plt.plot(knn_df['k'], knn_df['test_f1'], 's-', label='Test F1')\n",
    "plt.xlabel('n_neighbors (k)')\n",
    "plt.ylabel('F1 Score (macro)')\n",
    "plt.title('KNN: Pengaruh nilai k terhadap F1 Score')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.xticks(k_values)\n",
    "plt.show()\n",
    "\n",
    "# Tampilkan k terbaik\n",
    "best_k = knn_df.loc[knn_df['test_f1'].idxmax()]\n",
    "print(f\"\\nNilai k terbaik: {int(best_k['k'])} dengan Test F1 Score: {best_k['test_f1']:.4f}\")"
]))

# Decision boundaries for different k values
cells.append(make_code([
    "# Visualisasi Decision Boundary untuk beberapa nilai k\n",
    "fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for idx, k in enumerate(k_values):\n",
    "    knn_model = KNeighborsClassifier(n_neighbors=k)\n",
    "    knn_model.fit(X_train, y_train)\n",
    "    \n",
    "    DecisionBoundaryDisplay.from_estimator(\n",
    "        knn_model, X_test, response_method=\"predict\", \n",
    "        cmap='RdBu', alpha=0.3, grid_resolution=200, ax=axes[idx]\n",
    "    )\n",
    "    axes[idx].scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='w', s=20)\n",
    "    test_f1 = f1_score(y_test, knn_model.predict(X_test), average=\"macro\")\n",
    "    axes[idx].set_title(f'KNN k={k} (F1={test_f1:.3f})')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# Store best KNN in listmodel
cells.append(make_code([
    "# Simpan model KNN terbaik ke listmodel\n",
    "best_k_val = int(knn_df.loc[knn_df['test_f1'].idxmax(), 'k'])\n",
    "model = KNeighborsClassifier(n_neighbors=best_k_val)\n",
    "model.nickname = f\"KNN_k{best_k_val}_best\"\n",
    "model.fit(X_train, y_train)\n",
    "listmodel[model.nickname] = model\n",
    "\n",
    "pred = model.predict(X_test)\n",
    "print(f\"Best KNN (k={best_k_val}):\")\n",
    "print(\"Accuracy :\", accuracy_score(y_test, pred))\n",
    "print(\"F1 Score :\", f1_score(y_test, pred, average=\"macro\"))"
]))

# TODO answers for KNN section
cells.append(make_md([
    "<strong>TODO: </strong> \n",
    "1. Lakukan eksperimen dengan mengubah n_neighbors, berapa nilai n_neighbors yang memberikan hasil F1 score terbaik?\n",
    "2. Jelaskan bagaimana kemampuan kNN dalam melakukan klasifikasi multiclass dibandingkan ANN!\n",
    "3. Apa kekurangan KNN dibanding ANN?"
]))

cells.append(make_md([
    "**Jawaban:**\n",
    "\n",
    "**1. Nilai n_neighbors terbaik:**\n",
    "\n",
    "Berdasarkan eksperimen di atas, nilai `n_neighbors` yang memberikan F1 Score terbaik dapat dilihat dari tabel dan grafik. Umumnya, nilai k yang terlalu kecil (k=1) menyebabkan overfitting (train F1 sangat tinggi, test F1 lebih rendah), sedangkan k yang terlalu besar menyebabkan underfitting. Nilai k optimal biasanya berada di rentang 3-7 untuk dataset ini.\n",
    "\n",
    "**2. Kemampuan kNN vs ANN dalam Multiclass Classification:**\n",
    "\n",
    "- **kNN** secara natural mendukung multiclass classification tanpa modifikasi apapun. Algoritma ini bekerja dengan voting dari k tetangga terdekat, sehingga tidak peduli berapa banyak kelas yang ada.\n",
    "- **kNN** menghasilkan decision boundary yang sangat fleksibel dan non-linear, bahkan lebih detail dibanding ANN, karena boundary ditentukan langsung oleh distribusi data lokal.\n",
    "- **ANN** dengan aktivasi non-linear juga mampu menangani multiclass, namun memerlukan training/optimisasi parameter yang lebih kompleks.\n",
    "- Pada dataset ini, **kNN** cenderung memberikan performa yang kompetitif dibanding ANN, terutama karena dataset relatif kecil (500 sampel) dan hanya memiliki 2 fitur.\n",
    "\n",
    "**3. Kekurangan KNN dibanding ANN:**\n",
    "\n",
    "- **Kecepatan Prediksi:** kNN harus menghitung jarak ke semua data training setiap kali prediksi (lazy learning), sehingga sangat lambat pada dataset besar. ANN setelah training, prediksi hanya berupa forward pass yang sangat cepat.\n",
    "- **Skalabilitas:** kNN tidak cocok untuk dataset berdimensi tinggi (curse of dimensionality). ANN lebih mampu menangani feature space berdimensi tinggi.\n",
    "- **Memory:** kNN harus menyimpan seluruh dataset training di memori, sedangkan ANN hanya menyimpan bobot/parameter yang sudah dioptimisasi.\n",
    "- **Generalisasi:** ANN belajar representasi fitur internal melalui hidden layers, sehingga lebih mampu menggeneralisasi pola kompleks. kNN bergantung sepenuhnya pada metric jarak.\n",
    "- **Tidak ada model eksplisit:** kNN tidak menghasilkan model yang dapat diinterpretasi. ANN menghasilkan bobot yang bisa dianalisis.\n",
    "- **Sensitif terhadap noise:** kNN dengan k kecil sangat sensitif terhadap outlier dan noise. ANN lebih robust melalui proses training dan regularisasi."
]))

# ============================================================
# BUILD NOTEBOOK
# ============================================================

# Assign unique IDs to cells
import hashlib
for i, cell in enumerate(cells):
    cell_id = hashlib.md5(f"cell_{i}".encode()).hexdigest()[:8]
    cell["id"] = cell_id

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = r"d:\Muhammad 'Azmi Salam\Kuliah\Semester 4\Machine-Learning\Day-8-2 (kNN)\non-linearitas-ann-knn.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook berhasil dibuat!")
print(f"Output: {output_path}")
