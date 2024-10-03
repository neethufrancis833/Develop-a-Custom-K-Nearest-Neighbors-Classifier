from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


def find_euclidean_distance(a, b):
    return distance.euclidean(a, b)


# Load the Iris dataset
iris_dataset = datasets.load_iris()
features = iris_dataset.data
labels = iris_dataset.target

# Split the dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.5)


class KNNWithOneNeighbor:
    def fit(self, features_train, labels_train):
        self.features_train = features_train
        self.labels_train = labels_train

    def predict(self, features_test ):
        predictions = []
        for item in features_test:
            label = self.closest(item)
            predictions.append(label)
        return predictions

    def closest(self, item):
        best_distance = find_euclidean_distance(item, self.features_train[0])
        best_index = 0
        for i in range(1, len(self.features_train)):
            new_distance = find_euclidean_distance(item, self.features_train[i])
            if new_distance < best_distance:
                best_distance = new_distance
                best_index = i

        return self.labels_train[best_index]


# Create and fit the KNN classifier
classifier = KNNWithOneNeighbor()
classifier.fit(features_train, labels_train)
prediction = classifier.predict(features_test)

# Printing accuracy score
print(accuracy_score(labels_test, prediction))


# Predict the label for a new iris sample
new_iris_data = [[7.1, 2.9, 5.3, 2.4]]
iris_prediction = classifier.predict(new_iris_data)

# Output the predicted class
predicted_class = iris_prediction[0]
class_names = ['setosa', 'versicolor', 'virginica']
print(class_names[predicted_class])
