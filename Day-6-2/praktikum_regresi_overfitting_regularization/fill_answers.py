import json

path = "regresi_overfitting_regularization.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "JAWABAN 1" in src:
            cell["source"] = [
                "####PERTANYAAN 1\n",
                "1. Berapa jumlah data dalam dataset?\n",
                "2. Jika masalah yang akan diselesaikan adalah prediksi harga rumah, apa nama kolom labelnya?\n",
                "3. Berapa jumlah fiturnya (tidak termasuk kolom label)?\n",
                "4. Apakah range setiap fitur sama?\n",
                "\n",
                "JAWABAN 1 (Jawaban Singkat Saja):\n",
                "1. 1500\n",
                "2. harga_juta_rupiah\n",
                "3. 12\n",
                "4. Tidak, range setiap fitur bervariasi bergantung min dan max kolom tersebut."
            ]
        elif "JAWABAN 2" in src:
            cell["source"] = [
                "####PERTANYAAN 2\n",
                "1. Kenapa kita perlu melakukan Scaling?\n",
                "2. Apa dampaknya jika kita tidak melakukan scaling?\n",
                "\n",
                "JAWABAN 2:\n",
                "1. Agar semua fitur memiliki rentang nilai yang seragam, sehingga model konvergen lebih cepat dan fitur dengan rentang besar tidak mandominasi model.\n",
                "2. Proses training (konvergensi) akan lama dan bisa jadi model tidak mendapatkan bobot yang optimal karena error didominasi fitur berentang besar.\n"
            ]
        elif "JAWABAN 3" in src:
            cell["source"] = [
                "####PERTANYAAN 3\n",
                "1. Kenapa kita perlu melakukan split dataset menjaid training dan testing?\n",
                "2. Berapa jumlah data training?\n",
                "3. Berapa jumlah data testing?\n",
                "4. Berapa rasio data testing dibanding jml data keseluruhan?\n",
                "\n",
                "JAWABAN 3:\n",
                "1. Untuk mengevaluasi performa dan kemampuan generalisasi model pada data baru yang belum pernah dilihat sebelumnya.\n",
                "2. 1200\n",
                "3. 300\n",
                "4. 0.2 (20%)"
            ]
        elif "JAWABAN 4" in src:
            cell["source"] = [
                "####PERTANYAAN 4\n",
                "1. Berapa learning rate dan iterasi yang digunakan?\n",
                "2. Berapa Error memorization dan generalization yang didapat?\n",
                "3. Lakukan minimal 5 eksperimen untuk mendapatkan hasil terbaik, dengan mengubah jumlah iterasi dan learning rate, dokumentasikan hasilnya.\n",
                "4. Berapa Jml iterasi dan Learning Rate yang bisa memberikan hasil/MSE terbaik?\n",
                "\n",
                "JAWABAN 4:\n",
                "1. Learning rate 0.1 dan Iterasi 500\n",
                "2. Error Memorization (Train MSE): 726.99, Error Generalization (Test MSE): 757.57\n",
                "3. Tabel Hasil Eksperimen\n",
                "\n",
                "| Eksperimen | Jml Iterasi | Learning Rate | MSE Train | MSE Test |\n",
                "|------------|--------|---------------|-----|-----|\n",
                "| 1 | 500 | 0.01 | 6164.31 | 6403.56 |\n",
                "| 2 | 1000 | 0.01 | 4371.23 | 4560.57 |\n",
                "| 3 | 500 | 0.1 | 726.99 | 757.57 |\n",
                "| 4 | 1000 | 0.1 | 360.73 | 371.51 |\n",
                "| 5 | 500 | 0.5 | 315.11 | 321.12 |\n",
                "| 6 | 1000 | 0.5 | 313.51 | 319.17 |\n",
                "\n",
                "4. Jml Iterasi: 1000, Learning Rate: 0.5"
            ]
        elif "JAWABAN 5" in src:
            cell["source"] = [
                "####PERTANYAAN 5\n",
                "1. Berapa Error Memorization dan Generalization yang didapat?\n",
                "2. Bandingkan dengan ketika kita gunakan seluruh fiturnya, mana yang lebih baik?\n",
                "3. Lakukan eksperimen dengan menggunakan kombinasi fitur lainnya, seperti pada tabel berikut, dan catat hasilnya. Gunakan learning rate dan iterasi yang sama.\n",
                "4. Mana yang terbaik? apa kesimpulan Anda?\n",
                "5. Mengapa multiple feature dapat membantu?\n",
                "\n",
                "JAWABAN 5:\n",
                "1. Train MSE: 3785.76, Test MSE: 3788.91\n",
                "2. Menggunakan seluruh fitur jauh lebih baik (Train MSE: 313.51, Test MSE: 319.17).\n",
                "3. Tabel Hasil Eksperimen (dengan LR=0.5, Iters=1000)\n",
                "\n",
                "| Eksperimen | Fitur |  MSE Train | MSE Test |\n",
                "|------------|--------|---------------|-----|\n",
                "| 1 | seluruhnya | 313.51 | 319.17 |\n",
                "| 2 | 1 | 3785.76 | 3788.91 |\n",
                "| 3 | 2 | 7264.93 | 7773.02 |\n",
                "| 4 | 3 | 9783.20 | 9656.87 |\n",
                "| 6 | 1,2 | 871.24 | 976.89 |\n",
                "| 7 | 1,3 | 3416.93 | 3092.10 |\n",
                "| 8 | 2,3 | 6776.64 | 7543.52 |\n",
                "| 9 | 1,2,3 | 478.60 | 501.76 |\n",
                "\n",
                "4. Model dengan menggunakan seluruh fitur memberikan hasil terbaik. Kombinasi banyak fitur memberikan informasi harga yang lebih komprehensif.\n",
                "5. Multiple feature lebih baik karena merepresentasikan karakteristik target value (harga) dari sudut pandang yang berbeda, memperjelas batasan keputusan regresi dan mengidentifikasi pola kompleks yang tidak tertangkap oleh 1 fitur."
            ]
        elif "JAWABAN 6" in src:
            cell["source"] = [
                "####PERTANYAAN 6\n",
                "1. Berapa Error Memorization dan Generalization yang didapat pada poin b, c, dan d?\n",
                "2. Masalah apa yang terjadi pada poin b, d, kenapa hasilnya tidak sebaik pada poin c?\n",
                "3. Berdasarkan pengamatan Anda pada plot history training dan validation apa ciri-ciri gejala underfitting? bagaimana hubungannya dengan jml iterasi, jml data, dan jumlah fitur yang digunakan?\n",
                "4. Berdasarkan pengamatan Anda pada plot history training dan validation apa ciri-ciri gejala overfitting? bagaimana hubungannya dengan jml iterasi, jml data, dan jumlah fitur yang digunakan?\n",
                "\n",
                "JAWABAN 6:\n",
                "1. Error percobaan small data:\n\t- Poin b (1 fitur): Train = 8427.37, Test = 4398.03\n\t- Poin c (5 fitur): Train = 257.82, Test = 735.75\n\t- Poin d (Semua fitur): Train = 156.91, Test = 2108.68\n",
                "2. Poin b mengalami underfitting (error sangat tinggi meskipun di test data). Poin d mengalami overfitting (training kecil tapi hasil test besar). Poin c lebih ideal karena jumlah fitur cukup pas dengan jumlah data.\n",
                "3. Gejala underfitting: Loss training dan validation stabil di angka yang sangat tinggi. Berhubungan erat dengan model yang kurang kompleks (fitur terlalu sedikit), iterasi seringkali gagal konversikan error. \n",
                "4. Gejala overfitting: Loss training lambat laun terus rendah dan sangat kecil (hapal data), tapi validation errornya mandek dan malah bertambah besar lagi, jarak antar error ini semakin membesar (gap margin). Berhubungan saat model memiliki rasio fitur yang terlampau kompleks melampaui variasi dan jumlah baris data observasinya (small data)."
            ]
        elif "JAWABAN 7" in src:
            cell["source"] = [
                "####PERTANYAAN 7\n",
                "**Catatan**: Hanya ubah index fiturnya saja, jangan ubah parameter lain\n",
                "1. Lakukan sequential forward feature selection (SFS), untuk memilih fitur mana yang terbaik untuk digunakan, dengan cara melakukan training mulai dari 1 fitur, lalu menambahkan fitur yang paling meningkatkan performa model pada setiap langkah hingga semua fitur dicoba. Berdasarkan SFS kombinasi fitur mana yang terbaik?\n",
                "2. Lakukan sequential backward feature selection (SBFS), untuk memilih fitur mana yang terbaik untuk digunakan, dengan cara melakukan training mulai dari seluruh fitur, lalu menghapus fitur yang paling menurunkan performa model pada setiap langkah hingga semua fitur dicoba. Berdasarkan SBFS kombinasi fitur mana yang terbaik?\n",
                "\n",
                "JAWABAN 7:\n",
                "1. Kombinasi fitur SFS terbaik: [1, 2, 3, 4, 5, 7, 12, 8] dimana iterasi setelahnya malah membuat model menjadi buruk (tidak mengurangi validation MSE terbaik 552.06).\n",
                "2. Kombinasi fitur SBFS terbaik: Menghapus 8, 6, 11, 9, 10 menghasilkan validation MSE terbaik 566.12. Sisa fitur yang patut dipertahankan adalah: [1, 2, 3, 4, 5, 7, 12].\n"
            ]
        elif "JAWABAN 8" in src:
            cell["source"] = [
                "####PERTANYAAN 8\n",
                "**Catatan**: Hanya ubah index fiturnya saja, jangan ubah parameter lain\n",
                "1. Coba gunakan lamda= 0.0001, 0.001, 0.01, 0.1, 1, 10, 100. Mana yang memberikan hasil terbaik?\n",
                "2. Apa yang dapat Anda simpulkan?\n",
                "\n",
                "JAWABAN 8:\n",
                "1. Lamda = 0.01 memberikan hasil test generalisasi yang terbaik (Train MSE: 170.55, Test MSE: 2097.85) dibanding tidak menggunakan lamda sama sekali (Train: 156.91, Test: 2108.68).\n",
                "2. Regularisasi dapat mengurangi efek overfitting dengan menekan/memberikan penalti pada bobot parameter regresi berlebih tanpa membuangnya seutuhnya. Namun jika lamda terlalu besar, model menjauh dari pola data dan error akan semakin besar karena model berubah menjadi terlalu kaku (underfit)."
            ]
        elif "JAWABAN 9" in src:
            cell["source"] = [
                "####PERTANYAAN 9\n",
                "1. Berdasarkan grafik, apa kesimpulan Anda terhadap efek penambahan jumlah data terhadap kapasitas model?\n",
                "\n",
                "JAWABAN 9:\n",
                "1. Semakin banyak jumlah data yang tersedia atau dilatihkan, gap antara training error dan test error (generalization gap) semakin rapat/mengecil dan membaik. Penambahan jumlah data dapat membantu mencegah efek overfitting karena seiring besarnya kapasitas data model dapat berlatih keragaman data dengan jauh lebih baik tanpa mudah terjebak merapal memorisasi data sedikit.\n"
            ]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
