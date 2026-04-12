import 'dart:math';

class NaiveBayes {
  Map<int, double> _classProbabilities = {};
  Map<int, List<double>> _classMeans = {};
  Map<int, List<double>> _classVariances = {};

  void fit(List<List<double>> X, List<int> y) {
    if (X.length != y.length) {
      throw ArgumentError('X and y must have the same number of samples.');
    }

    int nSamples = X.length;
    int nFeatures = X[0].length;
    
    Map<int, List<List<double>>> classes = {};
    for (int i = 0; i < nSamples; i++) {
      classes.putIfAbsent(y[i], () => []).add(X[i]);
    }

    classes.forEach((c, samples) {
      _classProbabilities[c] = samples.length / nSamples;
      
      List<double> means = List.filled(nFeatures, 0.0);
      List<double> variances = List.filled(nFeatures, 0.0);

      // Calculate means
      for (var sample in samples) {
        for (int j = 0; j < nFeatures; j++) {
          means[j] += sample[j];
        }
      }
      for (int j = 0; j < nFeatures; j++) {
        means[j] /= samples.length;
      }

      // Calculate variances
      for (var sample in samples) {
        for (int j = 0; j < nFeatures; j++) {
          variances[j] += pow(sample[j] - means[j], 2);
        }
      }
      for (int j = 0; j < nFeatures; j++) {
        variances[j] /= samples.length;
        // Add a small epsilon to avoid division by zero
        variances[j] += 1e-9;
      }

      _classMeans[c] = means;
      _classVariances[c] = variances;
    });
  }

  int predict(List<double> x) {
    if (_classProbabilities.isEmpty) {
      throw StateError('Model is not trained yet. Call fit() first.');
    }

    int bestClass = -1;
    double maxPosterior = double.negativeInfinity;

    _classProbabilities.forEach((c, prior) {
      double posterior = log(prior);
      List<double> means = _classMeans[c]!;
      List<double> variances = _classVariances[c]!;

      for (int j = 0; j < x.length; j++) {
        posterior += _calculateGaussianLogProbability(x[j], means[j], variances[j]);
      }

      if (posterior > maxPosterior) {
        maxPosterior = posterior;
        bestClass = c;
      }
    });

    return bestClass;
  }

  double _calculateGaussianLogProbability(double x, double mean, double variance) {
    double exponent = exp(-pow(x - mean, 2) / (2 * variance));
    double prob = (1 / sqrt(2 * pi * variance)) * exponent;
    return log(max(prob, 1e-9)); // Prevent log(0)
  }
}
