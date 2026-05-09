class Perceptron {
  double learningRate;
  int epochs;
  late List<double> weights;
  double bias = 0.0;
  List<int> errorsPerEpoch = [];

  Perceptron({this.learningRate = 0.1, this.epochs = 20});

  int predict(List<double> x) {
    double linearOutput = _dot(x, weights) + bias;
    return linearOutput >= 0 ? 1 : 0;
  }

  void fit(List<List<double>> X, List<int> y) {
    int nFeatures = X[0].length;

    weights = List.filled(nFeatures, 0.0);
    bias = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
      int errors = 0;

      for (int i = 0; i < X.length; i++) {
        List<double> xi = X[i];
        int target = y[i];

        double linearOutput = _dot(xi, weights) + bias;
        int yPred = linearOutput >= 0 ? 1 : 0;

        double update = learningRate * (target - yPred);

        for (int j = 0; j < weights.length; j++) {
          weights[j] += update * xi[j];
        }

        bias += update;

        if (update != 0) {
          errors++;
        }
      }

      errorsPerEpoch.add(errors);
    }
  }

  double _dot(List<double> a, List<double> b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }
}