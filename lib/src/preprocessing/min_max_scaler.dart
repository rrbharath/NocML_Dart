class MinMaxScaler {
  List<double>? _mins;
  List<double>? _maxs;
  final double featureRangeMin;
  final double featureRangeMax;

  MinMaxScaler({this.featureRangeMin = 0.0, this.featureRangeMax = 1.0});

  void fit(List<List<double>> X) {
    if (X.isEmpty) return;

    int nFeatures = X[0].length;
    _mins = List.filled(nFeatures, double.infinity);
    _maxs = List.filled(nFeatures, double.negativeInfinity);

    for (var row in X) {
      for (int j = 0; j < nFeatures; j++) {
        if (row[j] < _mins![j]) _mins![j] = row[j];
        if (row[j] > _maxs![j]) _maxs![j] = row[j];
      }
    }
  }

  List<List<double>> transform(List<List<double>> X) {
    if (_mins == null || _maxs == null) {
      throw StateError('Scaler is not fitted yet. Call fit() first.');
    }

    List<List<double>> scaledX = List.generate(
      X.length,
      (i) => List.filled(X[0].length, 0.0),
    );

    for (int i = 0; i < X.length; i++) {
      for (int j = 0; j < X[i].length; j++) {
        double diff = _maxs![j] - _mins![j];
        if (diff == 0) {
          scaledX[i][j] = 0.0;
        } else {
          double std = (X[i][j] - _mins![j]) / diff;
          scaledX[i][j] = std * (featureRangeMax - featureRangeMin) + featureRangeMin;
        }
      }
    }

    return scaledX;
  }

  List<List<double>> fitTransform(List<List<double>> X) {
    fit(X);
    return transform(X);
  }
}
