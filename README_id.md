# NocML_Dart 🚀

*[Juga tersedia dalam Bahasa Inggris / Also available in English: [README.md](README.md)]*

Versi Dart & Flutter dari pustaka [NocML](https://github.com/Nocturnailed-Community/NocML), dikembangkan oleh **Nocturnailed Community**. Paket ini membawa kapabilitas machine learning yang ringan dan efisien secara langsung ke ekosistem Dart tanpa memerlukan integrasi rumit atau komponen eksternal berbasis C++.

[![Kreator](https://img.shields.io/badge/Creator-Muhammad--Ikhwan--Fathulloh-blue)](https://github.com/Muhammad-Ikhwan-Fathulloh)
[![Repositori](https://img.shields.io/badge/Repository-NocML__Dart-green)](https://github.com/Nocturnailed-Community/NocML_Dart)
[![Issues](https://img.shields.io/github/issues/Nocturnailed-Community/NocML_Dart?color=red)](https://github.com/Nocturnailed-Community/NocML_Dart/issues)

## 🌟 Gambaran Umum

`noc_ml_dart` pada dasarnya adalah hasil *porting* (translasi) dari mesin machine learning ringan kami yang pada awalnya ditujukan untuk mikrokontroler, yang sekarang telah ditulis secara idiomatik ke dalam Dart untuk pengembangan aplikasi murni yang bersifat lintas platform (Web, iOS, Android, Desktop).

Fitur-fitur:
- **Pra-pemrosesan Data (Preprocessing)**: StandardScaler, MinMaxScaler
- **Klasifikasi**: K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression
- **Klasterisasi (Clustering)**: K-Means
- **Peramalan / Regresi (Forecasting)**: Simple Linear Regression
- **Metrik Evaluasi**: Accuracy, MSE, MAE, R2-Score
- **Tanpa Dependensi (Zero Dependencies)**: Implementasi pure/murni menggunakan bahasa Dart

## 📦 Instalasi

Tambahkan baris berikut di dalam file `pubspec.yaml` Anda:

```yaml
dependencies:
  noc_ml_dart: ^1.0.0
```

Lalu jalankan `dart pub get` atau `flutter pub get`.

## 🚀 Mulai Cepat

Berikut adalah beberapa contoh cara mengintegrasikan `noc_ml_dart` pada proyek Anda.

### 1. Pra-pemrosesan Data

```dart
import 'package:noc_ml_dart/noc_ml_dart.dart';

void main() {
  List<List<double>> daftarDataMentah = [
    [10.0, 50.0],
    [20.0, 30.0],
    [30.0, 40.0]
  ];

  var scaler = StandardScaler();
  var dataSkala = scaler.fitTransform(daftarDataMentah);
  print(dataSkala);
}
```

### 2. K-Nearest Neighbors (KNN) & Evaluasi

```dart
import 'package:noc_ml_dart/noc_ml_dart.dart';

void main() {
  var knn = KNN(k: 3);
  
  // Data Latih: [fitur1, fitur2]
  List<List<double>> X_latih = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]];
  // Label/Kelas
  List<int> y_latih = [0, 0, 1, 1];
  
  knn.fit(X_latih, y_latih);
  
  // Lakukan Prediksi
  List<int> prediksi_y = [knn.predict([1.1, 1.9]), knn.predict([6.0, 9.0])];
  List<int> label_asli_y = [0, 1];
  
  // Evaluasi
  double akurasi = EvaluationMetrics.accuracyScore(label_asli_y, prediksi_y);
  print('Akurasi: $akurasi'); // Output: 1.0 (100%)
}
```

### 3. Peramalan (Simple Linear Regression)

```dart
import 'package:noc_ml_dart/noc_ml_dart.dart';

void main() {
  var regresi = SimpleLinearRegression();
  regresi.fit([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]);
  
  double hasilPeramalan = regresi.predict(6); 
  print('Hasil Ramalan Berikutnya: $hasilPeramalan'); // Output: 12.0
}
```

## 📚 Daftar Algoritma dan Modul yang Tersedia

### Preprocessing (Pra-pemrosesan)
- **StandardScaler**: Menstandardisasi fitur dengan cara menghilangkan rata-rata (`mean`) dan mengubah skalanya hingga ke dalam variansi satuan tunggal (menggunakan Z-score).
- **MinMaxScaler**: Mengubah fitur-fitur data Anda ke dalam rentang skala yang telah ditentukan (secara default: 0 hingga 1).

### Evaluation Metrics (Metrik Evaluasi)
- `accuracyScore()`: Melakukan kalkulasi persentase rasio dari total tebakan (prediksi) yang benar jika dibandingkan dengan total secara keseluruhan.
- `meanSquaredError()`: Rata-rata jarak simpangan pangkat dua antara angka yang diramalkan (prediksi) VS aktual aslinya (MSE).
- `meanAbsoluteError()`: Rata-rata dari nilai jarak simpangan mutlak (absolut) - MAE.
- `r2Score()`: Rasio koefisien determinasi dari model matematis dalam menyesuaikan pola tren (R-Squared).

### Classification (Klasifikasi)
- **Logistic Regression**: Menyediakan penyelesaian untuk klasifikasi tipe binari numerik menggunakan metode Gradient Descent.
- **Naive Bayes**: Metode pengelompokan secara probabilistik Gaussian yang bekerja dengan sangat cepat.
- **K-Nearest Neighbors (KNN)**: Mode yang tidak me-representasikan parameter di awal yang berbasis mencari letak jarak minimum untuk menentukan klasifikasi.

### Clustering (Klasterisasi)
- **K-Means**: Ini adalah logaritma yang tidak memerlukan panduan pendampingan (Unsupervised) sederhana tapi sangat praktis dalam menunjang algoritma yang bertugas menciptakan pemisahan atau *clustering*.

### Forecasting & Regression (Peramalan & Regresi)
- **Simple Linear Regression**: Algoritme yang dapat merepresenasikan estimasi perkiraan dengan menyusun tren regresi / garis linier lurus linear modeling.

## 🤝 Berkontribusi

Setiap kontribusi, pengajuan *issue*, maupun permintaan fitur tambahan sangatlah terbuka lebar! 
Silakan baca kolom isian perbaikan di [*issues page*](https://github.com/Nocturnailed-Community/NocML_Dart/issues).

## 📝 Lisensi

Pustaka/paket kode sumber publik ini berada di bawah otoritas lisensi perizinan kode terbuka versi standar [MIT](https://github.com/Nocturnailed-Community/NocML_Dart/blob/main/LICENSE).
