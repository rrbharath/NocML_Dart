import 'dart:math';

class StandardScaler {
  List<double>? _means;
  List<double>? _stds;

  void fit(List<List<double>> X) {
    if (X.isEmpty) return;

    int nSamples = X.length;
    int nFeatures = X[0].length;

    _means = List.filled(nFeatures, 0.0);
    _stds = List.filled(nFeatures, 0.0);

    // Calculate means
    for (int i = 0; i < nSamples; i++) {
      for (int j = 0; j < nFeatures; j++) {
        _means![j] += X[i][j];
      }
    }
    for (int j = 0; j < nFeatures; j++) {
      _means![j] /= nSamples;
    }

    // Calculate standard deviations
    for (int i = 0; i < nSamples; i++) {
      for (int j = 0; j < nFeatures; j++) {
        _stds![j] += pow(X[i][j] - _means![j], 2);
      }
    }
    for (int j = 0; j < nFeatures; j++) {
      _stds![j] = sqrt(_stds![j] / nSamples);
      if (_stds![j] == 0) {
        _stds![j] = 1.0; // avoid division by zero
      }
    }
  }

  List<List<double>> transform(List<List<double>> X) {
    if (_means == null || _stds == null) {
      throw StateError('Scaler is not fitted yet. Call fit() first.');
    }

    List<List<double>> scaledX = List.generate(
      X.length,
      (i) => List.filled(X[0].length, 0.0),
    );

    for (int i = 0; i < X.length; i++) {
      for (int j = 0; j < X[i].length; j++) {
        scaledX[i][j] = (X[i][j] - _means![j]) / _stds![j];
      }
    }

    return scaledX;
  }

  List<List<double>> fitTransform(List<List<double>> X) {
    fit(X);
    return transform(X);
  }
}
