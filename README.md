# NocML_Dart 🚀

*[Also available in Indonesian / Tersedia dalam Bahasa Indonesia: [README_id.md](README_id.md)]*

A Dart & Flutter version of the [NocML](https://github.com/Nocturnailed-Community/NocML) library, developed by the **Nocturnailed Community**. This package brings lightweight, efficient machine learning capabilities directly to Dart without requiring complex integrations or heavy external C++ components.

[![Creator](https://img.shields.io/badge/Creator-Muhammad--Ikhwan--Fathulloh-blue)](https://github.com/Muhammad-Ikhwan-Fathulloh)
[![Repository](https://img.shields.io/badge/Repository-NocML__Dart-green)](https://github.com/Nocturnailed-Community/NocML_Dart)
[![Issues](https://img.shields.io/github/issues/Nocturnailed-Community/NocML_Dart?color=red)](https://github.com/Nocturnailed-Community/NocML_Dart/issues)
[![Pub Version](https://img.shields.io/pub/v/noc_ml_dart)](https://pub.dev/packages/noc_ml_dart)

## 🌟 Overview

`noc_ml_dart` is designed strictly as a port of our lightweight machine learning engine geared initially toward microcontrollers, translated idiomatically to Dart for pure cross-platform development (Web, iOS, Android, Desktop).

Features:
- **Preprocessing**: StandardScaler, MinMaxScaler
- **Classification**: K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression
- **Clustering**: K-Means
- **Forecasting / Regression**: Simple Linear Regression
- **Evaluation Metrics**: Accuracy, MSE, MAE, R2-Score
- **Zero Dependencies**: Pure Dart implementation

## 📦 Installation

Add the following to your `pubspec.yaml` file:

```yaml
dependencies:
  noc_ml_dart: ^1.0.0
```

Then run `dart pub get` or `flutter pub get`.

## 🚀 Quick Start

Here are a few examples showcasing `noc_ml_dart`.

### 1. Preprocessing Data

```dart
import 'package:noc_ml_dart/noc_ml_dart.dart';

void main() {
  List<List<double>> rawData = [
    [10.0, 50.0],
    [20.0, 30.0],
    [30.0, 40.0]
  ];

  var scaler = StandardScaler();
  var scaledData = scaler.fitTransform(rawData);
  print(scaledData);
}
```

### 2. K-Nearest Neighbors (KNN) & Evaluation

```dart
import 'package:noc_ml_dart/noc_ml_dart.dart';

void main() {
  var knn = KNN(k: 3);
  
  // Training Data: [feature1, feature2]
  List<List<double>> X_train = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]];
  // Labels
  List<int> y_train = [0, 0, 1, 1];
  
  knn.fit(X_train, y_train);
  
  // Predict
  List<int> y_pred = [knn.predict([1.1, 1.9]), knn.predict([6.0, 9.0])];
  List<int> y_true = [0, 1];
  
  // Evaluate
  double acc = EvaluationMetrics.accuracyScore(y_true, y_pred);
  print('Accuracy: $acc'); // Output: 1.0 (100%)
}
```

### 3. Forecasting (Simple Linear Regression)

```dart
import 'package:noc_ml_dart/noc_ml_dart.dart';

void main() {
  var regression = SimpleLinearRegression();
  regression.fit([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]);
  
  double forecast = regression.predict(6); 
  print('Forecast: $forecast'); // Output: 12.0
}
```

## 📚 Supported Algorithms & Functions

### Preprocessing
- **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance (Z-score).
- **MinMaxScaler**: Transforms features by scaling each feature to a given range (default 0 to 1).

### Evaluation Metrics
- `accuracyScore()`: Ratio of correct predictions to total predictions.
- `meanSquaredError()`: Averages the squared differences between predicted and actual values.
- `meanAbsoluteError()`: Averages the absolute differences.
- `r2Score()`: Coefficient of determination indicating the proportion of the variance in the dependent variable.

### Classification
- **Logistic Regression**: Iterative gradient-descent binary classification.
- **Naive Bayes**: Fast Gaussian probabilistic classifier.
- **K-Nearest Neighbors (KNN)**: Distance-based classification.

### Clustering
- **K-Means**: Simple and effective unsupervised learning algorithm for partitioning data into k clusters.

### Forecasting & Regression
- **Simple Linear Regression**: Trend line prediction and standard linear modeling.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check [issues page](https://github.com/Nocturnailed-Community/NocML_Dart/issues).

## 📝 License

This project is [MIT](https://github.com/Nocturnailed-Community/NocML_Dart/blob/main/LICENSE) licensed.
