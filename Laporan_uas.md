# LAPORAN UAS KECERDASAN BUATAN

## Klasifikasi Rambu Lalu Lintas Menggunakan Convolutional Neural Network dan MobileNetV2

### Disusun oleh

| Nama | NIM |
|---|---:|
| Bambang Suryana G | 2406058 |
| Andika Rahman | 2406044 |

**Program Studi:** Teknik Informatika<br>
**Mata Kuliah:** Kecerdasan Buatan<br>
**Dosen Pengampu:** Dr. Leni Fitriani, S.T., M.Kom.<br>
**Domain proyek:** Computer Vision

---

## 1. Judul Proyek

**Klasifikasi Rambu Lalu Lintas Menggunakan Convolutional Neural Network (CNN) dan MobileNetV2**

Proyek ini membandingkan dua pendekatan *deep learning* untuk mengenali 43 kelas rambu lalu lintas pada German Traffic Sign Recognition Benchmark (GTSRB). CNN dibangun dari awal sebagai model dasar, sedangkan MobileNetV2 menggunakan *transfer learning* dengan bobot awal ImageNet.

## 2. Business Understanding

### 2.1 Permasalahan dunia nyata dan tinjauan literatur

Rambu lalu lintas menyampaikan batas kecepatan, larangan, peringatan, dan perintah yang harus dipahami pengguna jalan. Pada sistem bantuan pengemudi dan kendaraan cerdas, pengenalan rambu secara otomatis dibutuhkan agar sistem dapat memahami lingkungan jalan. Tantangannya meliputi perbedaan ukuran citra, sudut pandang, pencahayaan, kontras, kondisi fisik rambu, dan ketidakseimbangan jumlah sampel antarkelas.

GTSRB diperkenalkan sebagai masalah klasifikasi citra multikelas dengan lebih dari 50.000 citra rambu dan 43 kelas (Stallkamp et al., 2011; Stallkamp et al., 2012). CNN relevan karena dapat mempelajari fitur spasial secara hierarkis langsung dari citra (Krizhevsky et al., 2012; LeCun et al., 2015). MobileNetV2 menawarkan arsitektur yang lebih ringan melalui *inverted residuals* dan *linear bottlenecks*, sehingga menarik untuk dibandingkan dengan CNN konvensional (Sandler et al., 2018).

### 2.2 Tujuan proyek

1. Memahami karakteristik dan distribusi data GTSRB.
2. Menyiapkan citra agar sesuai untuk proses klasifikasi.
3. Membangun model CNN dan MobileNetV2 untuk klasifikasi 43 kelas.
4. Membandingkan kedua model menggunakan Accuracy, Precision, Recall, dan F1-Score.
5. Mengidentifikasi keterbatasan eksperimen dan memberikan rekomendasi perbaikan.

### 2.3 Pengguna sistem

Pengguna potensial sistem meliputi pengembang Advanced Driver Assistance Systems (ADAS), peneliti *computer vision*, pengembang kendaraan otonom, serta institusi pendidikan yang mempelajari klasifikasi citra.

### 2.4 Solusi dan manfaat implementasi AI

Solusi yang dirancang menerima citra rambu lalu lintas, melakukan praproses citra, kemudian menghasilkan prediksi salah satu dari 43 kelas. Implementasinya dapat membantu otomasi pengenalan rambu, menjadi komponen awal sistem bantuan pengemudi, serta menjadi bahan studi perbandingan antara model yang dilatih dari awal dan *transfer learning*.

## 3. Data Understanding

### 3.1 Sumber data

Dataset yang digunakan adalah **German Traffic Sign Recognition Benchmark (GTSRB)** versi Kaggle dari `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`. Notebook mengunduh data menggunakan `kagglehub.dataset_download()`.

### 3.2 Ukuran, format, dan struktur data

| Bagian | Jumlah | Keterangan |
|---|---:|---|
| Data pelatihan keseluruhan | 39.209 citra | Tersusun dalam folder kelas 0–42 |
| Data latih setelah pembagian | 31.368 citra | 80% dari folder `Train` |
| Data validasi | 7.841 citra | 20% dari folder `Train` |
| Data uji resmi | 12.630 citra | File citra di folder `Test`, label pada `Test.csv` |
| Jumlah kelas | 43 | Klasifikasi multikelas |

Citra menggunakan format PNG/JPG/JPEG dengan ukuran yang beragam. Seluruh citra diubah menjadi ukuran **224 × 224 piksel** dan tiga kanal warna agar sesuai dengan masukan kedua model.

### 3.3 Fitur dan target

| Elemen | Deskripsi |
|---|---|
| Fitur utama | Nilai piksel RGB pada citra rambu yang telah diubah ukurannya |
| `Width`, `Height` | Ukuran asli citra pada metadata data uji |
| `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2` | Koordinat *region of interest* pada metadata |
| `Path` | Lokasi file citra uji |
| `ClassId` | Label target berupa ID kelas 0–42 |

Target diubah menjadi representasi *one-hot* karena model menggunakan aktivasi Softmax dan *loss* `categorical_crossentropy`.

## 4. Exploratory Data Analysis (EDA)

### 4.1 Distribusi data

Notebook memvisualisasikan distribusi kelas menggunakan diagram batang dan diagram lingkaran. Statistik data latih adalah:

| Statistik | Nilai |
|---|---:|
| Total citra | 39.209 |
| Rata-rata citra per kelas | 911,84 |
| Jumlah minimum per kelas | 210 |
| Jumlah maksimum per kelas | 2.250 |
| Rasio maksimum/minimum | 10,71 |

### 4.2 Ketidakseimbangan kelas

Rasio 10,71 menunjukkan ketidakseimbangan kelas yang cukup jelas. Oleh sebab itu, akurasi tidak boleh menjadi satu-satunya dasar penilaian. Precision, Recall, dan F1-Score berbobot turut digunakan agar kontribusi setiap kelas dapat diamati lebih baik. Augmentasi juga digunakan untuk menambah variasi data latih, tetapi tidak secara langsung menyamakan jumlah setiap kelas.

### 4.3 Korelasi antarfitur

*Heatmap* dan *pairplot* tidak diterapkan karena masukan utama berupa tensor piksel berdimensi tinggi, bukan sekumpulan kecil fitur numerik tabular. Korelasi setiap piksel akan menghasilkan matriks yang sangat besar dan kurang interpretatif. Sebagai pengganti, EDA difokuskan pada ukuran citra, rasio aspek, distribusi intensitas piksel, distribusi kelas, dan contoh visual citra.

### 4.4 Insight pola data

1. Ukuran citra asli bervariasi, sehingga diperlukan *resize* yang konsisten.
2. Sebagian besar rasio aspek relatif seragam, sehingga risiko distorsi akibat *resize* dinilai terbatas.
3. Histogram intensitas menunjukkan variasi pencahayaan dan kontras.
4. Distribusi kelas tidak seimbang, dengan perbedaan jumlah sampel hingga 10,71 kali.
5. CNN dapat digunakan sebagai *baseline*, sedangkan MobileNetV2 sesuai untuk menguji manfaat *transfer learning* dan efisiensi parameter.

## 5. Data Preparation

### 5.1 Pembersihan data

Pemeriksaan notebook menemukan **39.209 nama file unik**, **0 duplikasi nama file**, dan **0 citra rusak**. Pada data citra, pemeriksaan nilai kosong dilakukan melalui validitas dan keterbacaan file, bukan pemeriksaan `null` seperti pada data tabular.

### 5.2 Encoding target

`flow_from_directory()` membaca nama folder sebagai kelas dan menghasilkan indeks kelas. Label kemudian dikembalikan sebagai vektor *one-hot*. Untuk data uji, `ClassId` pada `Test.csv` diubah menjadi teks dan dipasangkan dengan `Path` melalui *dataframe generator*.

### 5.3 Normalisasi dan augmentasi

Nilai piksel dinormalisasi dari rentang 0–255 menjadi 0–1 menggunakan `rescale=1./255`. Augmentasi data latih meliputi:

- rotasi hingga 20 derajat;
- pergeseran horizontal dan vertikal hingga 20%;
- *shear* hingga 0,15;
- *zoom* hingga 20%; dan
- pengisian area kosong dengan metode `nearest`.

*Horizontal flip* tidak digunakan karena dapat mengubah arti visual rambu tertentu.

### 5.4 Pembagian data

Folder `Train` dibagi menjadi 80% data latih (31.368 citra) dan 20% data validasi (7.841 citra). Data uji resmi sebanyak 12.630 citra dibaca melalui `Test.csv`. `shuffle=False` digunakan pada generator evaluasi agar urutan prediksi tetap selaras dengan label.

## 6. Modeling

### 6.1 Model CNN

CNN terdiri atas tiga blok Conv2D–BatchNormalization–MaxPooling, diikuti Flatten, Dense 256, Dropout 0,5, dan Dense 43 dengan Softmax. Model mempunyai **22.256.619 parameter**, dengan **22.256.171 parameter dapat dilatih**.

Kelebihan CNN adalah arsitekturnya sederhana dan dapat mempelajari karakteristik GTSRB dari awal. Kekurangannya adalah penggunaan `Flatten` setelah peta fitur 26 × 26 × 128 membuat lapisan Dense sangat besar dan meningkatkan biaya komputasi serta risiko *overfitting*.

### 6.2 Model MobileNetV2

MobileNetV2 dimuat dengan bobot ImageNet dan `include_top=False`. Seluruh *base model* dibekukan. Bagian klasifikasi terdiri atas GlobalAveragePooling2D, Dense 256, Dropout 0,5, dan Dense 43 dengan Softmax. Model mempunyai **2.596.971 parameter**, dengan **338.987 parameter dapat dilatih**.

Jumlah parameter MobileNetV2 sekitar 88,33% lebih sedikit dibandingkan CNN, sehingga lebih efisien untuk pelatihan dan berpotensi lebih sesuai bagi perangkat dengan sumber daya terbatas.

### 6.3 Konfigurasi kompilasi

Kedua model dikompilasi menggunakan optimizer Adam, `categorical_crossentropy`, dan metrik Accuracy.

### 6.4 Catatan implementasi penting

Pada notebook yang dianalisis, **tidak ditemukan pemanggilan `model.fit()` untuk CNN maupun MobileNetV2**. Artinya, lapisan klasifikasi yang baru dibuat belum dibuktikan telah menjalani proses pelatihan sebelum evaluasi. Hal ini merupakan penyebab paling kuat atas skor uji yang mendekati tebakan acak. Karena terdapat 43 kelas, peluang tebakan acak seragam sekitar 1/43 atau **2,33%**.

Selain itu, MobileNetV2 seharusnya menggunakan praproses yang sesuai dengan bobot ImageNet, misalnya `tf.keras.applications.mobilenet_v2.preprocess_input`. Eksperimen notebook hanya menggunakan `rescale=1./255`, sehingga distribusi masukan tidak sesuai dengan praproses bawaan MobileNetV2.

## 7. Evaluation

### 7.1 Confusion matrix

Notebook menghasilkan *confusion matrix* 43 × 43 untuk kedua model. Sebagian besar prediksi terkonsentrasi pada beberapa kelas dan banyak kelas mempunyai Precision serta Recall nol. Pola ini menunjukkan bahwa model belum mempelajari batas keputusan yang memadai.

### 7.2 Hasil metrik

Seluruh metrik berikut diambil dari output aktual notebook dan menggunakan rata-rata berbobot untuk Precision, Recall, serta F1-Score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| CNN | **0,0159 (1,59%)** | 0,0062 (0,62%) | **0,0159 (1,59%)** | 0,0041 (0,41%) |
| MobileNetV2 | 0,0147 (1,47%) | **0,0134 (1,34%)** | 0,0147 (1,47%) | **0,0095 (0,95%)** |

### 7.3 Penentuan model berdasarkan metrik

Tidak ada model yang dapat dinyatakan layak sebagai model klasifikasi akhir. CNN unggul tipis pada Accuracy dan Recall, sedangkan MobileNetV2 unggul pada Precision serta F1-Score. Jika aturan pemilihan hanya menggunakan Accuracy seperti pada kode notebook, CNN menjadi model dengan nilai tertinggi. Namun, selisih yang kecil dan skor absolut yang sangat rendah membuat pemilihan tersebut tidak bermakna secara praktis.

### 7.4 Inkonsistensi kesimpulan pada notebook

Salah satu sel markdown menyebut MobileNetV2 memperoleh Accuracy 98,74%, tetapi tidak terdapat output evaluasi yang mendukung angka itu. Output kode justru menunjukkan Accuracy MobileNetV2 1,47%. Karena itu, laporan ini menggunakan angka yang dapat diverifikasi dari output kode dan tidak menggunakan klaim 98,74%.

## 8. Kesimpulan dan Rekomendasi

### 8.1 Ringkasan hasil

Proyek berhasil melakukan EDA, menyiapkan generator data, membangun dua arsitektur, serta menjalankan evaluasi teknis pada 12.630 citra uji. Namun, tujuan untuk menghasilkan model klasifikasi rambu yang andal **belum tercapai**. CNN memperoleh Accuracy 1,59% dan MobileNetV2 1,47%, sehingga keduanya belum dapat digunakan dalam kondisi nyata.

### 8.2 Kelebihan

1. Menggunakan dataset benchmark dengan 43 kelas dan data uji resmi.
2. Melakukan pemeriksaan citra rusak dan duplikasi nama file.
3. Menyediakan EDA ukuran, rasio aspek, intensitas, dan distribusi kelas.
4. Membandingkan model dari awal dengan model *transfer learning*.
5. Menggunakan beberapa metrik dan *confusion matrix*.

### 8.3 Keterbatasan

1. Notebook tidak memuat proses `fit()` yang dapat diverifikasi.
2. Praproses MobileNetV2 tidak menggunakan `preprocess_input` yang sesuai.
3. Ketidakseimbangan kelas belum ditangani dengan *class weight* atau *resampling*.
4. Arsitektur CNN mempunyai lebih dari 22 juta parameter akibat penggunaan Flatten.
5. Tidak tersedia riwayat pelatihan, kurva *loss/accuracy*, jumlah epoch, atau model tersimpan.
6. Pemilihan model terbaik hanya berdasarkan Accuracy, padahal distribusi kelas tidak seimbang.

### 8.4 Rekomendasi perbaikan

1. Tambahkan proses pelatihan eksplisit menggunakan `model.fit()` dan simpan `history`.
2. Gunakan *early stopping*, *model checkpoint*, dan *learning-rate scheduler*.
3. Gunakan `mobilenet_v2.preprocess_input` untuk MobileNetV2 dan generator terpisah untuk CNN.
4. Setelah lapisan klasifikasi stabil, buka sebagian lapisan akhir MobileNetV2 untuk *fine-tuning* dengan *learning rate* kecil.
5. Pertimbangkan GlobalAveragePooling2D pada CNN untuk mengurangi jumlah parameter.
6. Tangani ketidakseimbangan dengan *class weight*, augmentasi terarah, atau *focal loss*.
7. Pastikan pemetaan indeks kelas data train, validation, dan test identik.
8. Laporkan metrik makro dan berbobot, *confusion matrix* ternormalisasi, serta performa per kelas.
9. Jalankan ulang seluruh notebook dari awal dalam satu sesi agar urutan eksekusi dan keluaran dapat direproduksi.

## 9. Referensi

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems, 25*, 1097–1105. https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436–444. https://doi.org/10.1038/nature14539

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520). https://doi.org/10.1109/CVPR.2018.00474

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In *2011 International Joint Conference on Neural Networks* (pp. 1453–1460). IEEE. https://doi.org/10.1109/IJCNN.2011.6033395

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks, 32*, 323–332. https://doi.org/10.1016/j.neunet.2012.02.016

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html

## 10. Lampiran

Lampiran utama proyek adalah `uas_model.ipynb`, yang memuat kode pengunduhan data, EDA, persiapan data, arsitektur model, *confusion matrix*, dan evaluasi. Dataset mentah tidak disimpan langsung di repositori karena ukurannya besar; petunjuk pengunduhan tersedia pada `README.md`.
