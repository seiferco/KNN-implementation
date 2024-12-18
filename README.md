# k-Nearest Neighbors Classifier

This project implements a k-Nearest Neighbors (kNN) classifier in Python. The kNN classifier predicts the class of a query point based on the labels of its `k` nearest neighbors in a dataset, as determined by Euclidean distance.

## Features

- **Efficient Neighbor Search**: Utilizes NumPy for fast computation of Euclidean distances and efficient indexing.
- **kNN Classification**: Predicts class labels for individual points and entire datasets.
- **Cross-Validation**: Implements K-fold cross-validation to evaluate model performance.
- **Hyperparameter Search**: Tests different values of `k` to find the best-performing hyperparameter.
- **Kaggle Submission**: Generates predictions for a test set in the correct format for submission.

## Code Overview

### Main Functions

1. **`get_nearest_neighbors(example_set, query, k)`**
   - Finds the indices of the `k` nearest neighbors for a given query point.
   - Uses NumPy's broadcasting and `np.linalg.norm` for efficient computation.

2. **`knn_classify_point(examples_X, examples_y, query, k)`**
   - Classifies a single query point based on the majority label of its `k` nearest neighbors.

3. **`cross_validation(train_X, train_y, num_folds, k)`**
   - Implements K-fold cross-validation to compute the average accuracy and variance of the kNN classifier across different folds.

4. **`compute_accuracy(true_y, predicted_y)`**
   - Calculates the fraction of correct predictions compared to the true labels.

5. **`predict(examples_X, examples_y, queries_X, k)`**
   - Predicts class labels for all points in a query dataset using the kNN classifier.

### Execution Flow

1. **Sanity Checks**:
   - Validates the correctness of the kNN implementation using example data.

2. **Hyperparameter Search**:
   - Performs a grid search over different `k` values using 4-fold cross-validation.

3. **Kaggle Submission**:
   - Generates predictions for the test dataset and saves them in a CSV file (`test_predicted.csv`) in the required format.

### Input/Output

#### Input Files:
- `train.csv`: Training data with features and labels.
- `test_pub.csv`: Test data with features.

#### Output Files:
- `test_predicted.csv`: Predicted labels for the test dataset in the format required for Kaggle submissions.

### Example

#### Running the Script

```bash
python knn_classifier.py
```

#### Sample Output
```
k =     1 -- train acc = 95.83%  val acc = 84.50% (0.0012)      [exe_time = 0.32]
k =     3 -- train acc = 92.45%  val acc = 86.75% (0.0010)      [exe_time = 0.40]
k =     5 -- train acc = 90.12%  val acc = 87.10% (0.0015)      [exe_time = 0.52]
...
```

#### Generated CSV Format

```csv
id,income
0,1
1,0
2,1
...
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install dependencies:
   ```bash
   pip install numpy
   ```

3. Place the `train.csv` and `test_pub.csv` files in the working directory.

## License

This project is open-source and available under the MIT License. Feel free to use and modify it as needed!
