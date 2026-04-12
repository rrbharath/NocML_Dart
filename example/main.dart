import 'package:noc_ml_dart/noc_ml_dart.dart';

void main() {
  print('--- NocML Dart Examples ---');

  // 0. Preprocessing Example
  print('\n[0] Preprocessing (Standard Scaler)');
  var scaler = StandardScaler();
  List<List<double>> rawData = [
    [10, 50],
    [20, 30],
    [30, 40]
  ];
  var scaledData = scaler.fitTransform(rawData);
  print('Raw: $rawData\nScaled: $scaledData');

  // 1. KNN Classification Example
  print('\n[1] KNN Classification');
  var knn = KNN(k: 3);
  List<List<double>> X_train = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
  ];
  List<int> y_train = [0, 0, 1, 1, 0, 1];
  knn.fit(X_train, y_train);
  int knnPred = knn.predict([8.0, 9.0]);
  print('KNN Predicted Class for [8.0, 9.0]: $knnPred (Expected: 1)');

  // 1.5 Evaluation Example
  List<int> y_true_clf = [1];
  List<int> y_pred_clf = [knnPred];
  print(
      'Current KNN Accuracy: ${EvaluationMetrics.accuracyScore(y_true_clf, y_pred_clf)}');

  // 2. Logistic Regression Example
  print('\n[2] Logistic Regression');
  var lr = LogisticRegression(learningRate: 0.1, iterations: 1000);
  lr.fit(X_train, y_train);
  int lrPred = lr.predict([1.0, 1.0]);
  print('LR Predicted Class for [1.0, 1.0]: $lrPred (Expected: 0)');

  // 3. Naive Bayes Example
  print('\n[3] Naive Bayes Classification');
  var nb = NaiveBayes();
  nb.fit(X_train, y_train);
  int nbPred = nb.predict([5.0, 8.0]);
  print('NB Predicted Class for [5.0, 8.0]: $nbPred (Expected: 1)');

  // 4. KMeans Clustering Example
  print('\n[4] KMeans Clustering');
  var kmeans = KMeans(k: 2, maxIterations: 100);
  List<List<double>> X_cluster = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
  ];
  kmeans.fit(X_cluster);
  int clusterId = kmeans.predict([8.0, 9.0]);
  print('Point [8.0, 9.0] belongs to cluster: $clusterId');
  print('Centroids are: ${kmeans.centroids}');

  // 5. Simple Linear Regression Example
  print('\n[5] Simple Linear Regression (Forecasting)');
  var slr = SimpleLinearRegression();
  List<double> X_fore = [1, 2, 3, 4, 5];
  List<double> y_fore = [2, 4, 5, 4, 5];
  slr.fit(X_fore, y_fore);
  double forecast = slr.predict(6);
  print('Forecasted value for x=6: ${forecast.toStringAsFixed(2)}');

  List<double> y_pred_reg = X_fore.map((x) => slr.predict(x)).toList();
  print(
      'Regression MSE: ${EvaluationMetrics.meanSquaredError(y_fore, y_pred_reg)}');
  print(
      'Regression R2 Score: ${EvaluationMetrics.r2Score(y_fore, y_pred_reg)}');
}
