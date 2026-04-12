import 'package:noc_ml_dart/noc_ml_dart.dart';
import 'package:test/test.dart';

void main() {
  group('Classification Tests', () {
    test('KNN prediction', () {
      var knn = KNN(k: 3);
      List<List<double>> X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      List<int> y = [0, 0, 1, 1, 0, 1];
      knn.fit(X, y);
      expect(knn.predict([8, 9]), equals(1));
      expect(knn.predict([1, 1]), equals(0));
    });

    test('Logistic Regression prediction', () {
      var lr = LogisticRegression(learningRate: 0.1, iterations: 1000);
      List<List<double>> X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      List<int> y = [0, 0, 1, 1, 0, 1];
      lr.fit(X, y);
      expect(lr.predict([8, 9]), equals(1));
      expect(lr.predict([1, 1]), equals(0));
    });

    test('Naive Bayes prediction', () {
      var nb = NaiveBayes();
      List<List<double>> X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      List<int> y = [0, 0, 1, 1, 0, 1];
      nb.fit(X, y);
      expect(nb.predict([8, 9]), equals(1));
      expect(nb.predict([1, 1]), equals(0));
    });
  });

  group('Clustering Tests', () {
    test('KMeans fit and predict', () {
      var kmeans = KMeans(k: 2);
      List<List<double>> X = [[1, 2], [2, 1], [8, 9], [9, 8]];
      kmeans.fit(X);
      
      int c1 = kmeans.predict([1, 1]);
      int c2 = kmeans.predict([9, 9]);
      
      expect(c1, isNot(equals(c2)));
      expect(kmeans.centroids.length, equals(2));
    });
  });

  group('Forecasting Tests', () {
    test('Simple Linear Regression fit and predict', () {
      var slr = SimpleLinearRegression();
      List<double> X = [1, 2, 3, 4, 5];
      List<double> y = [2, 4, 6, 8, 10]; // Exact linear: y = 2x
      slr.fit(X, y);
      
      expect(slr.slope, closeTo(2.0, 0.001));
      expect(slr.intercept, closeTo(0.0, 0.001));
      expect(slr.predict(6), closeTo(12.0, 0.001));
    });
  });

  group('Preprocessing Tests', () {
    test('StandardScaler', () {
      var scaler = StandardScaler();
      List<List<double>> X = [[10], [20], [30]];
      var Xt = scaler.fitTransform(X);
      
      // Mean is 20, Std is sqrt(((10-20)^2 + 0 + (30-20)^2)/3) = sqrt(200/3) = 8.1649
      expect(Xt[0][0], closeTo((10-20)/8.1649658, 0.001));
      expect(Xt[1][0], closeTo(0, 0.001));
      expect(Xt[2][0], closeTo((30-20)/8.1649658, 0.001));
    });

    test('MinMaxScaler', () {
      var scaler = MinMaxScaler();
      List<List<double>> X = [[10], [20], [30]];
      var Xt = scaler.fitTransform(X);
      expect(Xt[0][0], equals(0.0));
      expect(Xt[1][0], equals(0.5));
      expect(Xt[2][0], equals(1.0));
    });
  });

  group('Metrics Tests', () {
    test('Accuracy Score', () {
      expect(EvaluationMetrics.accuracyScore([1,0,1], [1,0,0]), closeTo(0.666, 0.01));
    });
    
    test('Mean Squared Error', () {
      expect(EvaluationMetrics.meanSquaredError([1,2,3], [1,2,2]), equals(1.0/3.0));
    });

    test('R2 Score', () {
      // Data: 1, 2, 3 -> Mean = 2
      // ssTot = (-1)^2 + 0 + 1^2 = 2
      // Pred: 1, 2, 2 -> Res = 0 + 0 + (3-2)^2 = 1
      // r2 = 1 - (1/2) = 0.5
      expect(EvaluationMetrics.r2Score([1, 2, 3], [1, 2, 2]), equals(0.5));
    });
  });
}
