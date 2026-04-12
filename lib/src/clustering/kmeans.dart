import 'dart:math';

class KMeans {
  final int k;
  final int maxIterations;
  List<List<double>>? _centroids;

  KMeans({this.k = 3, this.maxIterations = 100});

  void fit(List<List<double>> X) {
    if (X.isEmpty) {
      throw ArgumentError('Input data cannot be empty.');
    }

    int nSamples = X.length;
    int nFeatures = X[0].length;

    if (k > nSamples) {
      throw ArgumentError(
          'Number of clusters (k) cannot be greater than the number of samples.');
    }

    // Initialize centroids randomly from the data points
    _centroids = [];
    var random = Random();
    Set<int> initialIndices = {};
    while (initialIndices.length < k) {
      initialIndices.add(random.nextInt(nSamples));
    }
    for (int index in initialIndices) {
      _centroids!.add(List.from(X[index]));
    }

    for (int iteration = 0; iteration < maxIterations; iteration++) {
      // Assign samples to nearest centroid
      List<List<List<double>>> clusters = List.generate(k, (_) => []);
      for (int i = 0; i < nSamples; i++) {
        int nearestCentroid = _getNearestCentroid(X[i]);
        clusters[nearestCentroid].add(X[i]);
      }

      // Update centroids
      List<List<double>> newCentroids =
          List.generate(k, (_) => List.filled(nFeatures, 0.0));
      for (int i = 0; i < k; i++) {
        if (clusters[i].isEmpty) {
          // If a cluster is empty, keep the old centroid to prevent division by zero
          newCentroids[i] = List.from(_centroids![i]);
          continue;
        }

        for (var sample in clusters[i]) {
          for (int j = 0; j < nFeatures; j++) {
            newCentroids[i][j] += sample[j];
          }
        }

        for (int j = 0; j < nFeatures; j++) {
          newCentroids[i][j] /= clusters[i].length;
        }
      }

      // Check for convergence
      bool converged = true;
      for (int i = 0; i < k; i++) {
        if (_euclideanDistance(_centroids![i], newCentroids[i]) > 1e-4) {
          converged = false;
          break;
        }
      }

      _centroids = newCentroids;

      if (converged) {
        break;
      }
    }
  }

  int predict(List<double> x) {
    if (_centroids == null) {
      throw StateError('Model is not trained yet. Call fit() first.');
    }
    return _getNearestCentroid(x);
  }

  List<List<double>> get centroids {
    if (_centroids == null) {
      throw StateError('Model is not trained yet.');
    }
    return _centroids!;
  }

  int _getNearestCentroid(List<double> x) {
    int nearest = -1;
    double minDistance = double.infinity;

    for (int i = 0; i < k; i++) {
      double distance = _euclideanDistance(x, _centroids![i]);
      if (distance < minDistance) {
        minDistance = distance;
        nearest = i;
      }
    }

    return nearest;
  }

  double _euclideanDistance(List<double> a, List<double> b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
  }
}
