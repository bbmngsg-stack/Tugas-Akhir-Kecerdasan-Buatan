# Klasifikasi Rambu Lalu Lintas — CNN vs MobileNetV2

Repositori UAS Kecerdasan Buatan untuk membandingkan **Convolutional Neural Network (CNN)** dan **MobileNetV2** pada klasifikasi 43 kelas rambu lalu lintas dari **German Traffic Sign Recognition Benchmark (GTSRB)**.

## Anggota Kelompok
| Bambang Suryana G | 2406058 |
| Andika Rahman | 2406044 |

**Program Studi:** Teknik Informatika<br>
**Mata Kuliah:** Kecerdasan Buatan<br>
**Dosen Pengampu:** Dr. Leni Fitriani, S.T., M.Kom.

## Ringkasan Proyek

Alur proyek mencakup:

1. pengunduhan dan pemahaman dataset GTSRB;
2. EDA distribusi kelas, ukuran, rasio aspek, dan intensitas piksel;
3. pemeriksaan duplikasi nama file dan citra rusak;
4. normalisasi, augmentasi, encoding label, dan pembagian data;
5. pembangunan model CNN dan MobileNetV2;
6. evaluasi menggunakan *confusion matrix*, Accuracy, Precision, Recall, dan F1-Score; serta
7. analisis keterbatasan dan rekomendasi perbaikan.

## Struktur Repositori

```text
UAS-KecerdasanBuatan/
├── README.md
├── Laporan_uas.md
├── uas_model.ipynb
└── data/
    ├── dataset/
    └── Jurnal/
```


## Dataset

- **Nama:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Sumber Kaggle:** `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`
- **Jenis:** klasifikasi citra multikelas
- **Jumlah kelas:** 43
- **Data train:** 39.209 citra
- **Data test resmi:** 12.630 citra

Dataset diunduh dari notebook menggunakan `kagglehub`, sehingga file mentah tidak harus dimasukkan ke GitHub.

## Cara Menjalankan

### Opsi A — Google Colab

1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih **File → Upload notebook**.
3. Unggah `uas_model.ipynb`.
4. Aktifkan GPU melalui **Runtime → Change runtime type → T4 GPU**.
5. Jalankan sel instalasi dependensi berikut jika pustaka belum tersedia:

   ```python
   !pip install -q kagglehub tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn pillow pydot graphviz
   ```

6. Jalankan semua sel secara berurutan melalui **Runtime → Run all**.

### Opsi B — Komputer Lokal

Persyaratan yang disarankan:

- Python 3.10 atau 3.11;
- Jupyter Notebook/JupyterLab;
- RAM minimal 8 GB, disarankan 16 GB; dan
- GPU yang kompatibel bersifat opsional, tetapi sangat membantu proses pelatihan.

Buat lingkungan virtual dan instal dependensi:

```bash
python -m venv .venv
```

Aktifkan pada Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instal pustaka:

```bash
pip install jupyter kagglehub tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn pillow pydot graphviz
```

Jalankan notebook:

```bash
jupyter notebook uas_model.ipynb
```

## Urutan Pengerjaan

### 1. Import library

Memuat NumPy, Pandas, OpenCV, Matplotlib, TensorFlow/Keras, dan Scikit-learn.

### 2. Mengunduh dataset

Notebook memanggil:

```python
dataset_path = kagglehub.dataset_download(
    "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
)
```

### 3. Data understanding dan EDA

Memeriksa struktur folder, jumlah kelas, distribusi citra per kelas, ketidakseimbangan kelas, contoh gambar, ukuran, rasio aspek, dan histogram intensitas piksel.

### 4. Data preparation

- memeriksa citra rusak dan duplikasi nama file;
- melakukan *one-hot encoding* label;
- mengubah ukuran citra menjadi 224 × 224;
- menormalisasi nilai piksel;
- menerapkan augmentasi; dan
- membagi data train–validation sebesar 80:20.

Data uji dibaca dari `Test.csv` karena folder `Test` tidak mempunyai subfolder kelas.

### 5. Modeling

- **CNN:** tiga blok konvolusi, Dense 256, Dropout, dan Softmax 43 kelas.
- **MobileNetV2:** bobot ImageNet, *base model* dibekukan, GlobalAveragePooling2D, Dense 256, Dropout, dan Softmax 43 kelas.

### 6. Evaluation

Evaluasi dilakukan pada 12.630 citra uji menggunakan Accuracy, weighted Precision, weighted Recall, weighted F1-Score, *classification report*, dan *confusion matrix*.

## Hasil Eksperimen yang Terekam

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| CNN | 1,59% | 0,62% | 1,59% | 0,41% |
| MobileNetV2 | 1,47% | 1,34% | 1,47% | 0,95% |

Hasil tersebut **belum layak digunakan sebagai model akhir**. CNN unggul tipis dalam Accuracy dan Recall, sedangkan MobileNetV2 unggul dalam Precision dan F1-Score.

## Catatan Reproduksibilitas dan Perbaikan Wajib

Notebook yang tersedia belum memperlihatkan pemanggilan `model.fit()` untuk kedua model. Jika langsung dievaluasi setelah kompilasi, lapisan klasifikasi masih menggunakan bobot acak dan hasilnya akan sangat rendah. Sebelum menjadikan hasil sebagai kesimpulan akhir:

1. tambahkan dan jalankan proses pelatihan kedua model;
2. gunakan generator praproses terpisah untuk CNN dan MobileNetV2;
3. gunakan `mobilenet_v2.preprocess_input` untuk MobileNetV2;
4. simpan riwayat pelatihan dan tampilkan grafik Accuracy/Loss;
5. gunakan `EarlyStopping` dan `ModelCheckpoint`;
6. pastikan pemetaan indeks kelas train dan test sama; dan
7. jalankan ulang seluruh notebook dari awal tanpa melewati sel.

Contoh minimum proses pelatihan:

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

cnn_history = cnn_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=[early_stopping]
)
```

MobileNetV2 perlu dilatih dengan generator yang menggunakan praproses khusus MobileNetV2, bukan sekadar `rescale=1./255`.

## Dokumen

Laporan lengkap tersedia di [`Laporan_uas.md`](Laporan_uas.md). Laporan memuat Business Understanding, Data Understanding, EDA, Data Preparation, Modeling, Evaluation, kesimpulan, rekomendasi, dan minimal lima referensi ilmiah bergaya APA.

## Referensi Utama

1. Stallkamp et al. (2011), *The German Traffic Sign Recognition Benchmark*. https://doi.org/10.1109/IJCNN.2011.6033395
2. Stallkamp et al. (2012), *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. https://doi.org/10.1016/j.neunet.2012.02.016
3. Sandler et al. (2018), *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. https://doi.org/10.1109/CVPR.2018.00474
4. Krizhevsky et al. (2012), *ImageNet Classification with Deep Convolutional Neural Networks*. https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
5. LeCun et al. (2015), *Deep Learning*. https://doi.org/10.1038/nature14539
