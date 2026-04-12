class SimpleLinearRegression {
  double? _slope;
  double? _intercept;

  void fit(List<double> X, List<double> y) {
    if (X.length != y.length) {
      throw ArgumentError('X and y must have the same number of samples.');
    }

    int n = X.length;
    double sumX = 0;
    double sumY = 0;
    double sumXY = 0;
    double sumXX = 0;

    for (int i = 0; i < n; i++) {
      sumX += X[i];
      sumY += y[i];
      sumXY += X[i] * y[i];
      sumXX += X[i] * X[i];
    }

    _slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    _intercept = (sumY - _slope! * sumX) / n;
  }

  double predict(double x) {
    if (_slope == null || _intercept == null) {
      throw StateError('Model is not trained yet. Call fit() first.');
    }
    return _slope! * x + _intercept!;
  }

  double get slope {
    if (_slope == null) {
      throw StateError('Model is not trained yet.');
    }
    return _slope!;
  }

  double get intercept {
    if (_intercept == null) {
      throw StateError('Model is not trained yet.');
    }
    return _intercept!;
  }
}
