from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd


# Load data and encode variables with hot encoding
def load_data_hot_encoding(url):
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    dataset = pd.read_csv(url, names=names)
    names.remove('class')
    dataset = pd.get_dummies(dataset, columns=names, prefix=names)
    classEncode = {'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}}
    dataset.replace(classEncode, inplace=True)    
    return dataset


# Create validation and test sets
def create_datasets(data):
    array = data.values
    x = array[:,1:]    # Variables
    y = array[:,0]     # Class name (Label)
    return train_test_split(x, y, test_size=0.20)


# Creates model and prints performance metrics
def create_model(data):
    X_train, X_test, y_train, y_test = create_datasets(data)
    model = SVC()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


data = load_data_hot_encoding('car.csv')
create_model(data)

