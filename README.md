# Diabetes Severity Classification menggunakan Fuzzy C-Means dan Neural Network

## Ringkasan Singkat
Proyek ini menggabungkan Fuzzy C-Means (FCM) untuk clustering kelembaman (soft clustering) dengan model Neural Network berbasis PyTorch untuk melakukan klasifikasi dan penilaian tingkat keparahan diabetes pasien. Hasilnya berupa derajat keanggotaan fuzzy untuk setiap cluster, skor severity kontinu, serta analisis performa model dan profil pasien yang komprehensif.

## Kegunaan, Fungsi, dan Manfaat
- Kegunaan: Menentukan kategori severity (derajat keparahan) pada pasien berdasarkan fitur klinis (mis. Glucose, BMI, Age) dengan pendekatan fuzzy + supervised learning.
- Fungsi utama:
  - Pra-pemrosesan data (normalisasi, pembagian dataset).
  - Clustering Fuzzy C-Means untuk menghasilkan membership fuzzy dan pusat cluster.
  - Pelatihan model Neural Network yang output-nya diinterpretasikan sebagai membership soft (softmax).
  - Penghitungan skor severity dari membership (baik FCM maupun NN).
  - Evaluasi performa (accuracy, AUC, confusion matrix, laporan metrik).
  - Eksport hasil (CSV, gambar, laporan teks) dan pembuatan profil pasien.
- Manfaat:
  - Memberikan estimasi risiko yang lebih halus (bukan sekadar biner).
  - Memungkinkan interpretasi pasien melalui membership ke beberapa cluster.
  - Analisis performa yang lengkap untuk pengambilan keputusan klinis atau penelitian lanjutan.

## Struktur Proyek (peta folder utama)
- main.py                     -> Orkestrasi pipeline (pra-proses, FCM, pelatihan, evaluasi, analisis)
- config.yaml                 -> Konfigurasi eksperimen (path data, hyperparameter FCM dan model)
- src/
  - data_preprocessing.py     -> Kelas DiabetesDataPreprocessor (load, preprocess, split, save scaler)
  - fcm_clustering.py         -> Kelas FuzzyCMeans (fit, predict_cluster, calculate_severity_scores, visualize, save_centers)
  - torch_model.py            -> Definisi model PyTorch (EnhancedDiabetesClassifier) dan loss (RegularizedFuzzyLoss, FocalLoss)
  - train.py                  -> EnhancedModelTrainer (dataloader, training loop, evaluasi, plotting, laporan)
- data/
  - processed/                -> Data yang sudah di-normalisasi (disimpan oleh pipeline)
- models/                     -> Salinan model terlatih (disimpan saat training)
- results/
  - fcm_results/              -> Hasil cluster FCM (cluster assignments, visualisasi)
  - model_performance/        -> Training history, confusion matrix, ROC, detailed_metrics_report.txt, metrics_report.txt
  - severity_analysis/        -> comprehensive_patient_profiles.csv dan analisis severity lainnya

> Contoh file keluaran yang ada di repository:
> - results/model_performance/detailed_metrics_report.txt (mengandung precision/recall/f1, accuracy, AUC, confusion matrix)
> - results/fcm_results/cluster_assignments.csv
> - results/severity_analysis/comprehensive_patient_profiles.csv

## Penjelasan Fungsi Modul (singkat)
- DiabetesDataPreprocessor (src/data_preprocessing.py)
  - load_data(): Membaca CSV dari path di `config.yaml`, menampilkan shape dan distribusi kelas.
  - preprocess(): Normalisasi/scaling, pembuatan fitur tambahan jika ada.
  - split_data(): Membagi data menjadi train/val/test stratified.
  - save_scaler(): Menyimpan objek scaler ke disk.
- FuzzyCMeans (src/fcm_clustering.py)
  - fit(X_train): Menjalankan FCM pada data training, mengembalikan pusat cluster dan matrix membership.
  - predict_cluster(X): Menghitung membership untuk data baru (val/test).
  - calculate_severity_scores(membership): Mengkonversi membership menjadi skor severity (mis. bobot cluster -> normalisasi ke [0,1]).
  - visualize_clusters(...): Membuat plot cluster (jika memungkinkan) dan menyimpannya ke `results/fcm_results`.
  - save_centers(): Menyimpan pusat cluster ke disk.
- EnhancedDiabetesClassifier, RegularizedFuzzyLoss, FocalLoss (src/torch_model.py)
  - Model NN dengan beberapa blok, batchnorm, inisialisasi bobot, dan fungsi forward yang mengeluarkan logits membership.
  - predict_with_softmax(x): Menghasilkan membership (softmax) untuk input.
  - Loss khusus menggabungkan regularisasi fuzzy dan focal loss untuk menekan overfitting dan menangani imbalance.
- EnhancedModelTrainer (src/train.py)
  - prepare_dataloaders(...): Membuat DataLoader PyTorch, memperhitungkan sample weighting bila imbalance.
  - train(...): Loop training dengan validasi, early stopping, menyimpan model terbaik.
  - evaluate(...): Menghitung metrik di test set, menghasilkan probabilitas, prediksi.
  - plot_confusion_matrix(), plot_roc_curve(), plot_training_history(): Menyimpan grafik evaluasi.
  - generate_detailed_report(): Menulis laporan metrik ke `results/model_performance/detailed_metrics_report.txt`.

## Alur / Step-by-step (sesuai main.py)
1. Persiapan folder
   - `main.py` membuat folder yang diperlukan: `data/processed`, `models`, `results/fcm_results`, `results/model_performance`, `results/severity_analysis`.
2. Load konfigurasi
   - Membaca `config.yaml` (parameter FCM, model, path data, dsb).
3. Step 1 — Data Preprocessing
   - Load dataset dari `config.yaml`.
   - Cek missing values, cek distribusi kelas.
   - Normalisasi fitur (StandardScaler) -> disimpan ke `data/processed/normalized_data.csv`.
   - Split data ke train/val/test (stratified).
   - Simpan scaler untuk penggunaan inferensi nanti.
4. Step 2 — Fuzzy C-Means (FCM)
   - Jalankan FCM pada data training (`fcm.fit(X_train)`), dapatkan `centers` dan `membership`.
   - Hitung severity scores untuk training (`calculate_severity_scores`).
   - Simpan hasil cluster di `results/fcm_results/cluster_assignments.csv`.
   - (Opsional) Visualisasi cluster disimpan di `results/fcm_results/`.
5. Step 3 — Prediksi membership FCM untuk validation & test
   - Hitung membership untuk X_val dan X_test (`predict_cluster`).
   - Hitung severity untuk val & test.
6. Step 4 — Persiapan dan Pelatihan NN (Enhanced)
   - Buat dataloader yang menggabungkan fitur X dan membership FCM bila digunakan.
   - Inisialisasi model `EnhancedDiabetesClassifier` (output dim = jumlah cluster FCM).
   - Latih model dengan `EnhancedModelTrainer.train()`, gunakan teknik anti-overfitting (regularisasi, weighted sampling, focal loss, dsb).
   - Simpan model terbaik ke folder `models/`.
7. Step 5 — Evaluasi Model
   - Hitung prediksi, probabilitas, metrik: accuracy, AUC, precision/recall/f1.
   - Simpan plot training history, confusion matrix, ROC curve di `results/model_performance/`.
   - Tulis laporan metrik di `results/model_performance/detailed_metrics_report.txt`.
8. Step 6 — Analisis Severity dan Profil Pasien
   - Untuk seluruh pasien (train+val+test), ambil prediksi membership NN dan skor severity NN.
   - Gabungkan dengan membership FCM dan skor severity FCM.
   - Klasifikasikan kategori risiko (Low, Medium, High, Critical) berdasarkan bins skor.
   - Simpan `comprehensive_patient_profiles.csv` ke `results/severity_analysis/`.
9. Step 7 — Ringkasan Akhir
   - Tampilkan ringkasan jumlah pasien, proporsi diabetic/healthy, jumlah cluster FCM, metrik terbaik, dsb.
   - Contoh hasil yang ditemukan di repo: Accuracy sekitar 84.4%, AUC ~0.9165 (lihat `detailed_metrics_report.txt`).

## Cara Install & Menjalankan (contoh)
1. Clone repo
   - git clone https://github.com/ardhikaxx/diabetes-fcm.git
   - cd diabetes-fcm
2. (Disarankan) Buat virtual environment
   - python -m venv .venv
   - source .venv/bin/activate  (Linux/macOS) atau .venv\Scripts\activate (Windows)
3. Install dependency (contoh umum)
   - pip install -r requirements.txt
   Jika file requirements.txt tidak tersedia, install paket minimal:
   - pip install numpy pandas scikit-learn matplotlib seaborn pyyaml torch tqdm
4. Siapkan file konfigurasi `config.yaml`
   - Pastikan `data.path` menunjuk ke dataset CSV (contoh: dataset diabetes dengan kolom `Outcome`, `Glucose`, `BMI`, `Age`, dsb.)
   - Contoh parameter penting:
     - fcm:
       - n_clusters: 3
       - m: 2.0
       - max_iter: 150
       - error: 1e-5
     - model:
       - hidden_dim: 64
       - learning_rate: 0.001
       - epochs: 100
5. Jalankan pipeline utama
   - python main.py
   - Program akan menjalankan seluruh pipeline (preprocessing -> FCM -> training NN -> evaluasi -> analisis severity) dan menyimpan hasil di folder `results/` dan `models/`.

## Konfigurasi (ringkasan)
- `config.yaml` mengendalikan:
  - Path dataset (data.path)
  - Hyperparameter FCM (n_clusters, m, max_iter, error)
  - Hyperparameter model (hidden_dim, learning_rate, batch_size, epochs)
  - Opsi training dan regularisasi
- Ubah parameter di `config.yaml` untuk eksperimen (mis. lebih banyak cluster, ubah ukuran hidden, dsb).

## Output yang Diharapkan & Letak File
- data/processed/normalized_data.csv
- results/fcm_results/cluster_assignments.csv
- results/fcm_results/cluster_visualization.png (jika berhasil)
- models/* (model terlatih disimpan di folder ini)
- results/model_performance/training_history.png
- results/model_performance/confusion_matrix.png
- results/model_performance/roc_curve.png
- results/model_performance/detailed_metrics_report.txt
- results/severity_analysis/comprehensive_patient_profiles.csv

## Catatan Praktis & Tips
- Pastikan dataset memiliki kolom `Outcome` sebagai label biner (0 = healthy, 1 = diabetic).
- Jika dataset imbalance, trainer sudah menggunakan teknik weighted sampling dan focal loss untuk membantu.
- Visualisasi cluster kadang memerlukan reduksi dimensi — jika fitur banyak, fungsi visualize mungkin gagal; handling pengecualian sudah ada di `main.py`.
- Simpan konfigurasi eksperimen dan versi model agar hasil dapat direproduksi.