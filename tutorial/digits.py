import os
# Suppress unwanted messages from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Or any {'0', '1', '2'}
# Using cpu due to gpu running out of memory (core dumped)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# Make a prediction for a new image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# Load train and test dataset
def load_dataset():
    # Load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # Reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # Encode handwritten numbers to integer
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# Scale pixels
def prep_pixels(train, test):
    # Convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # Normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


# Define cnn model
# All layers will use ReLu activation function and the He weight
# initialization function
def define_model():
    model = Sequential()    # Sequential model (stack layers sequentially after each other)
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))  # Convolutional layer with 3x3 filter. 32 filters in total
    model.add(MaxPooling2D((2, 2)))  # Max pooling layer
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())  # Provides features to the classifier
    # Dense layer to interpret the features
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))  # Layer to give 10 output nodes
    # Compile model
    # Search technique used to update weights (Stochastic Gradient Descent
    # with support for momentum)
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


# Evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # Prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # Enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # Define model
        model = define_model()
        # Select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # Fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32,
                            validation_data=(testX, testY), verbose=0)
        # Evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # Stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# Plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # Plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(
            histories[i].history['val_loss'],
            color='orange',
            label='test')
        # Plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(
            histories[i].history['accuracy'],
            color='blue',
            label='train')
        pyplot.plot(
            histories[i].history['val_accuracy'],
            color='orange',
            label='test')
    pyplot.show()


# Summarize model performance
def summarize_performance(scores):
    # Print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(scores) * 100, std(scores) * 100, len(scores)))
    # Box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# Create and save a model
def create_and_save_model(filename):
    # Load dataset
    trainX, trainY, testX, testY = load_dataset()
    # Prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # Define model
    model = define_model()
    # Fit model
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    # Save model
    model.save(filename)


# Run the test harness for evaluating a model
def run_test_harness(filename):
    # Load dataset
    trainX, trainY, testX, testY = load_dataset()
    # Prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # Load model
    model = load_model(filename)
    # Evaluate model on test dataset
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))

    # Evaluate model
    #scores, histories = evaluate_model(trainX, trainY)
    # Learning curves
    # summarize_diagnostics(histories)
    # Summarize estimated performancer
    # summarize_performance(scores)


# Load and prepare the image
def load_image(filename):
	# Load the image
	img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
	# Convert to array
	img = img_to_array(img)
	# Reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# Prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


# Load an image and predict the class
def run_example():
	# Load the image
	img = load_image('sample_image.png')
	# Load model
	model = load_model('digits_model.h5')
	# Predict the class
	digit = model.predict_classes(img)
	print(digit[0])


run_example()
#run_test_harness('digits_model.h5')
