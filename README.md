# Develop-a-Custom-K-Nearest-Neighbors-Classifier

Language: Python (version 3.12.5)	

IDE: PyCharm

Installation Steps
---------------------
1. Install Python 3 and PyCharm
2. Install scikit-learn - “pip install scikit-learn”
   
Troubleshooting Installation Issues
-------------------------------------

If you encounter any issues while installing scikit-learn, try the following alternative method:

1. Open PyCharm and navigate to File > Settings.
2. Select Project from the left sidebar, then click on Project Interpreter.
3. Click the "+" button to add a new package.
4. Search for "scikit-learn," select it, and click Install.

Project Description
--------------------
In this project, I developed a custom K-Nearest Neighbors classifier that focuses on a single nearest neighbor for classification. The dataset used is the Iris dataset.

Iris Dataset
-------------
The Iris dataset, introduced by Ronald A. Fisher in 1936, is a foundational resource in machine learning. It consists of 150 samples of iris flowers, categorized into three species:

1. Iris setosa
2. Iris versicolor
3. Iris virginica
   
Each sample includes four key features:
----------------------------------------
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

Code Explanation
----------------
1. Import Required Modules   (datasets, train_test_split,distance,accuracy_score)
2. Create a function to calculate the Euclidean distance between two data points.
3. Load the Iris dataset and extract the features and target labels.
4. Split the Dataset.Use "train_test_split" method to divide the dataset into training and testing sets. Here  test_size is  
    0.5, means that 50% of the data will be used for training and 50% for testing. The dataset contains 150 samples.
5. Define a class for the K-Nearest Neighbors classifier, including three methods:
   **fit()**: Trains the model using the training data.
   **predict()**: Predicts the output for the test data.
   **closest()**: Finds the nearest neighbor to a given data point.   
6. After creating the classifier, train it using the features_train and labels_train data.
7. Use new_iris_data as the test data or new sample for classification. This data point will be passed to the predict() 
   method to determine its species/label based on the trained model.

<img width="960" alt="knn_classifier" src="https://github.com/user-attachments/assets/5b4672f7-a757-4028-9616-0a3916c28b23">




