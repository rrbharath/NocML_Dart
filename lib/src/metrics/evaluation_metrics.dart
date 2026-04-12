import 'dart:math';

class EvaluationMetrics {
  /// Classification: Accuracy Score
  static double accuracyScore(List<int> yTrue, List<int> yPred) {
    if (yTrue.length != yPred.length) {
      throw ArgumentError('yTrue and yPred must have the same length.');
    }

    int correct = 0;
    for (int i = 0; i < yTrue.length; i++) {
      if (yTrue[i] == yPred[i]) {
        correct++;
      }
    }
    return correct / yTrue.length;
  }

  /// Regression/Forecasting: Mean Squared Error
  static double meanSquaredError(List<double> yTrue, List<double> yPred) {
    if (yTrue.length != yPred.length) {
      throw ArgumentError('yTrue and yPred must have the same length.');
    }

    double sum = 0;
    for (int i = 0; i < yTrue.length; i++) {
      sum += pow(yTrue[i] - yPred[i], 2);
    }
    return sum / yTrue.length;
  }

  /// Regression/Forecasting: Mean Absolute Error
  static double meanAbsoluteError(List<double> yTrue, List<double> yPred) {
    if (yTrue.length != yPred.length) {
      throw ArgumentError('yTrue and yPred must have the same length.');
    }

    double sum = 0;
    for (int i = 0; i < yTrue.length; i++) {
      sum += (yTrue[i] - yPred[i]).abs();
    }
    return sum / yTrue.length;
  }

  /// Regression/Forecasting: R-squared (Coefficient of Determination)
  static double r2Score(List<double> yTrue, List<double> yPred) {
    if (yTrue.length != yPred.length) {
      throw ArgumentError('yTrue and yPred must have the same length.');
    }

    double meanY = yTrue.reduce((a, b) => a + b) / yTrue.length;
    double ssTot = 0;
    double ssRes = 0;

    for (int i = 0; i < yTrue.length; i++) {
      ssTot += pow(yTrue[i] - meanY, 2);
      ssRes += pow(yTrue[i] - yPred[i], 2);
    }

    if (ssTot == 0) return 1.0;
    return 1 - (ssRes / ssTot);
  }
}
