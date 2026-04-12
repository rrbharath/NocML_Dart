import 'dart:math';

class LogisticRegression {
  final double learningRate;
  final int iterations;
  List<double>? _weights;
  double _bias = 0.0;

  LogisticRegression({this.learningRate = 0.01, this.iterations = 1000});

  void fit(List<List<double>> X, List<int> y) {
    if (X.length != y.length) {
      throw ArgumentError('X and y must have the same number of samples.');
    }

    int nSamples = X.length;
    int nFeatures = X[0].length;

    _weights = List.filled(nFeatures, 0.0);
    _bias = 0.0;

    for (int i = 0; i < iterations; i++) {
      List<double> dw = List.filled(nFeatures, 0.0);
      double db = 0.0;

      for (int j = 0; j < nSamples; j++) {
        double linearModel = _dotProduct(X[j], _weights!) + _bias;
        double yPredicted = _sigmoid(linearModel);

        for (int k = 0; k < nFeatures; k++) {
          dw[k] += (yPredicted - y[j]) * X[j][k];
        }
        db += (yPredicted - y[j]);
      }

      for (int k = 0; k < nFeatures; k++) {
        _weights![k] -= learningRate * dw[k] / nSamples;
      }
      _bias -= learningRate * db / nSamples;
    }
  }

  int predict(List<double> x) {
    if (_weights == null) {
      throw StateError('Model is not trained yet. Call fit() first.');
    }

    double linearModel = _dotProduct(x, _weights!) + _bias;
    double yPredicted = _sigmoid(linearModel);

    return yPredicted >= 0.5 ? 1 : 0;
  }

  double _dotProduct(List<double> a, List<double> b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  double _sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
  }
}
