# Machine Learning Fundamentals

## What is Machine Learning?
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Instead of following fixed rules, ML algorithms build mathematical models based on training data.

## Types of Machine Learning

### Supervised Learning
The algorithm learns from labeled training data with known input-output pairs.

**Common Algorithms:**
- Linear Regression: Predicts continuous values
- Logistic Regression: Binary classification
- Decision Trees: Tree-based decisions
- Random Forest: Ensemble of decision trees
- Support Vector Machines (SVM): Classification with margins
- Neural Networks: Deep learning models

**Use Cases:**
- Email spam detection
- House price prediction
- Image classification
- Sentiment analysis
- Medical diagnosis

### Unsupervised Learning
The algorithm finds patterns in unlabeled data without predefined outputs.

**Common Algorithms:**
- K-Means Clustering: Groups similar data points
- Hierarchical Clustering: Creates cluster hierarchies
- DBSCAN: Density-based clustering
- PCA (Principal Component Analysis): Dimensionality reduction
- Autoencoders: Neural network for feature learning

**Use Cases:**
- Customer segmentation
- Anomaly detection
- Recommendation systems
- Data compression
- Market basket analysis

### Reinforcement Learning
The algorithm learns through trial and error, receiving rewards or penalties.

**Key Concepts:**
- Agent: The learner or decision maker
- Environment: What the agent interacts with
- State: Current situation
- Action: What the agent can do
- Reward: Feedback from the environment

**Use Cases:**
- Game playing (Chess, Go, video games)
- Robotics
- Autonomous vehicles
- Resource management
- Trading strategies

## Key Concepts

### Training and Testing
- **Training Set**: Data used to train the model (typically 70-80%)
- **Validation Set**: Data used to tune hyperparameters (10-15%)
- **Test Set**: Data used to evaluate final model (10-15%)

### Overfitting and Underfitting
- **Overfitting**: Model learns training data too well, including noise
  - High training accuracy, low test accuracy
  - Solutions: Regularization, more data, simpler model

- **Underfitting**: Model is too simple to capture patterns
  - Low training and test accuracy
  - Solutions: More complex model, more features, less regularization

### Bias-Variance Tradeoff
- **High Bias**: Model is too simple (underfitting)
- **High Variance**: Model is too complex (overfitting)
- **Goal**: Find the sweet spot between bias and variance

## Evaluation Metrics

### Classification Metrics
```python
# Accuracy: Correct predictions / Total predictions
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision: True positives / Predicted positives
precision = TP / (TP + FP)

# Recall (Sensitivity): True positives / Actual positives
recall = TP / (TP + FN)

# F1 Score: Harmonic mean of precision and recall
f1_score = 2 * (precision * recall) / (precision + recall)
```

### Regression Metrics
```python
# Mean Absolute Error
MAE = (1/n) * Σ|y_actual - y_predicted|

# Mean Squared Error
MSE = (1/n) * Σ(y_actual - y_predicted)²

# Root Mean Squared Error
RMSE = √MSE

# R² Score (Coefficient of Determination)
R² = 1 - (SS_res / SS_tot)
```

## Feature Engineering

### Feature Scaling
- **Standardization**: (x - mean) / std_dev (z-score normalization)
- **Normalization**: (x - min) / (max - min) (scales to 0-1)
- **Log Transformation**: log(x) for skewed distributions

### Feature Selection
- **Filter Methods**: Statistical tests (correlation, chi-square)
- **Wrapper Methods**: Use model performance (forward/backward selection)
- **Embedded Methods**: Built into algorithms (Lasso, Ridge)

### Handling Missing Data
- **Deletion**: Remove rows or columns with missing values
- **Imputation**: Fill with mean, median, mode, or predicted values
- **Indicator**: Add binary column indicating missingness

## Common ML Algorithms

### Linear Regression
Simple model for predicting continuous values:
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Assumptions:**
- Linear relationship between features and target
- Independence of errors
- Homoscedasticity (constant error variance)
- Normally distributed errors

### Logistic Regression
Binary classification algorithm:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
```

**Output**: Probability between 0 and 1 using sigmoid function

### Decision Trees
Tree-based model making sequential decisions:
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Advantages:**
- Easy to interpret
- Handles non-linear relationships
- Requires little data preprocessing

**Disadvantages:**
- Prone to overfitting
- Unstable (small changes in data can change tree structure)

### Random Forest
Ensemble of decision trees:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
feature_importance = model.feature_importances_
```

**Key Parameters:**
- n_estimators: Number of trees
- max_depth: Maximum tree depth
- min_samples_split: Minimum samples to split node

### Support Vector Machines
Finds optimal hyperplane for classification:
```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Kernels:**
- Linear: For linearly separable data
- RBF (Radial Basis Function): Most common, handles non-linear data
- Polynomial: For polynomial relationships

### K-Means Clustering
Unsupervised algorithm for grouping data:
```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels_
centers = model.cluster_centers_
```

**Algorithm:**
1. Initialize k random centroids
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat steps 2-3 until convergence

## Neural Networks

### Basic Architecture
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
```

### Common Activation Functions
- **ReLU**: f(x) = max(0, x) - Most common in hidden layers
- **Sigmoid**: f(x) = 1/(1+e^-x) - Binary classification output
- **Tanh**: f(x) = (e^x - e^-x)/(e^x + e^-x) - Alternative to sigmoid
- **Softmax**: Multi-class classification output

### Optimization Algorithms
- **SGD (Stochastic Gradient Descent)**: Basic optimizer
- **Adam**: Adaptive learning rate, most popular
- **RMSprop**: Good for recurrent neural networks
- **AdaGrad**: Adapts learning rate per parameter

## Best Practices

### Data Preparation
1. Explore and visualize data
2. Handle missing values
3. Encode categorical variables
4. Scale numerical features
5. Split into train/validation/test sets
6. Address class imbalance if present

### Model Development
1. Start with simple baseline model
2. Try multiple algorithms
3. Perform cross-validation
4. Tune hyperparameters systematically
5. Use ensemble methods
6. Monitor for overfitting

### Model Evaluation
1. Use appropriate metrics for your problem
2. Evaluate on held-out test set
3. Analyze errors and edge cases
4. Check for fairness and bias
5. Test on real-world data when possible

### Deployment Considerations
1. Model versioning
2. Monitoring model performance
3. Retraining strategy
4. Inference speed requirements
5. Resource constraints
6. Explainability requirements

## Common Pitfalls

1. **Data Leakage**: Using test data in training
2. **Not Shuffling Data**: Biased train/test splits
3. **Ignoring Class Imbalance**: Poor minority class performance
4. **Feature Scaling Issues**: Forgetting to scale features
5. **Overfitting**: Too complex model for data size
6. **Wrong Metric**: Optimizing for wrong business goal
7. **Not Using Cross-Validation**: Unreliable performance estimates
8. **Ignoring Domain Knowledge**: Missing important features
9. **Correlation vs Causation**: Confusing the two
10. **Not Handling Outliers**: Skewed model performance

## Learning Resources
- Online courses: Coursera, edX, fast.ai
- Books: "Hands-On Machine Learning" by Aurélien Géron
- Practice: Kaggle competitions
- Documentation: Scikit-learn, TensorFlow, PyTorch
- Papers: arXiv.org for latest research
