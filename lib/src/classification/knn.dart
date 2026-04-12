import 'dart:math';

class KNN {
  final int k;
  List<List<double>>? _X_train;
  List<int>? _y_train;

  KNN({this.k = 3});

  void fit(List<List<double>> X, List<int> y) {
    if (X.length != y.length) {
      throw ArgumentError('X and y must have the same number of samples.');
    }
    _X_train = X;
    _y_train = y;
  }

  int predict(List<double> x) {
    if (_X_train == null || _y_train == null) {
      throw StateError('Model is not trained yet. Call fit() first.');
    }

    List<Map<String, dynamic>> distances = [];

    for (int i = 0; i < _X_train!.length; i++) {
      double dist = _euclideanDistance(x, _X_train![i]);
      distances.add({'distance': dist, 'label': _y_train![i]});
    }

    distances.sort(
        (a, b) => (a['distance'] as double).compareTo(b['distance'] as double));

    Map<int, int> classCounts = {};
    for (int i = 0; i < k && i < distances.length; i++) {
      int label = distances[i]['label'] as int;
      classCounts[label] = (classCounts[label] ?? 0) + 1;
    }

    int predictedLabel = -1;
    int maxCount = -1;
    classCounts.forEach((label, count) {
      if (count > maxCount) {
        maxCount = count;
        predictedLabel = label;
      }
    });

    return predictedLabel;
  }

  double _euclideanDistance(List<double> a, List<double> b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
  }
}
