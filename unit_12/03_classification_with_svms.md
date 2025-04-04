# Classification with Support Vector Machines (SVMs)

## Understanding SVMs: The Geometric Intuition

Support Vector Machines (SVMs) are primarily binary classifiers that work by finding the optimal separation boundary (hyperplane) between two classes of data. The key concept is finding the maximum margin between classes.

### Basic Concept: The Hyperplane

A hyperplane is a separation boundary whose dimension depends on the space it's dividing:
- In 1D space: A point (0D) separates a line
- In 2D space: A line (1D) separates a plane
- In 3D space: A plane (2D) separates volume
- In nD space: A hyperplane ((n-1)D) separates the space

```python
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def visualize_2d_svm():
    # Create sample data (e.g., business vs sports documents)
    X = np.array([
        [1, 2], [2, 3], [2, 1],  # Class 1 (e.g., business)
        [6, 5], [7, 8], [8, 7]   # Class 2 (e.g., sports)
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Train SVM
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    
    # Plot decision boundary
    plt.scatter(X[:3, 0], X[:3, 1], label='Business')
    plt.scatter(X[3:, 0], X[3:, 1], label='Sports')
    
    # Draw hyperplane
    w = svm.coef_[0]
    b = svm.intercept_[0]
    x_points = np.linspace(0, 10, 100)
    y_points = -(w[0] * x_points + b) / w[1]
    
    plt.plot(x_points, y_points, 'k-', label='Maximum Margin Hyperplane')
    plt.legend()
    plt.title('SVM Document Classification')
    plt.show()
```

## Handling Non-Linear Cases: Kernel Tricks

Sometimes data isn't linearly separable in its original space. SVMs use kernel tricks to transform data into higher dimensions where it becomes separable.

### 1. The Mod Function Transform
```python
def mod_kernel_example():
    """
    Transform 1D non-separable data to 2D separable data using mod function
    """
    # Original 1D data (e.g., numbers 1-8)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Odd vs Even
    
    # Transform to 2D using mod function
    X_transformed = np.column_stack([X, X % 2])
    return X_transformed, y
```

### 2. The Square Transform
```python
def square_kernel_example():
    """
    Transform data using square function for circular patterns
    """
    # Generate circular pattern
    center = np.random.normal(0, 1, (50, 2))  # Center cluster
    outer = np.random.normal(0, 4, (50, 2))   # Outer points
    
    X = np.vstack([center, outer])
    y = np.hstack([np.zeros(50), np.ones(50)])
    
    # Transform using squared distances
    X_transformed = np.column_stack([
        X,
        np.sum(X**2, axis=1)  # Add third dimension
    ])
    return X_transformed, y
```

## Practical Document Classification with SVM

```python
from sklearn.feature_extraction.text import TfidfVectorizer

class DocumentSVMClassifier:
    def __init__(self, kernel='linear'):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = SVC(kernel=kernel)
        
    def fit(self, documents, labels):
        """
        Train the SVM classifier on document data
        """
        # Transform documents to TF-IDF features
        X = self.vectorizer.fit_transform(documents)
        self.classifier.fit(X, labels)
        
    def predict(self, documents):
        """
        Classify new documents
        """
        X = self.vectorizer.transform(documents)
        return self.classifier.predict(X)

# Example usage
documents = [
    "The stock market showed strong gains today",
    "The team won the championship game",
    "Quarterly earnings exceeded expectations",
    "The player scored in the final minute"
]
labels = ["business", "sports", "business", "sports"]

classifier = DocumentSVMClassifier()
classifier.fit(documents, labels)

new_doc = ["Company shares rose 5% after the announcement"]
prediction = classifier.predict([new_doc])
```

## Handling Outliers and Noise

SVMs can handle outliers through soft margins, controlled by the C parameter:

```python
class RobustSVMClassifier:
    def __init__(self, C=1.0):
        self.classifier = SVC(
            C=C,           # Lower C allows more violations
            kernel='rbf',  # Radial basis function for non-linear patterns
            probability=True
        )
        
    def fit_with_confidence(self, X, y):
        """
        Train classifier and identify potential outliers
        """
        self.classifier.fit(X, y)
        # Get distances from hyperplane
        distances = abs(self.classifier.decision_function(X))
        # Flag potential outliers
        potential_outliers = distances < np.percentile(distances, 10)
        return potential_outliers
```

## Best Practices for SVM Document Classification

1. **Data Preparation**
   - Normalize features
   - Handle imbalanced classes
   - Remove noise and outliers

2. **Kernel Selection**
   - Linear: For high-dimensional text data
   - RBF: For complex patterns
   - Custom: For specific domain knowledge

3. **Parameter Tuning**
   - C: Controls margin violations
   - Gamma: Influences decision boundary complexity
   - Kernel parameters: Specific to each kernel type

4. **Performance Optimization**
   - Feature selection
   - Dimensionality reduction
   - Efficient text preprocessing

The power of SVMs lies in their ability to:
- Find optimal separation boundaries
- Handle non-linear relationships through kernel tricks
- Work effectively with high-dimensional text data
- Provide robust classification even with outliers 