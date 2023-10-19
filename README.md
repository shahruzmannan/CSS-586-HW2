# MLP-and-NNs

## Logistic regression

I implemented Logistic Regression with Mini-batch Gradient Descent using TensorFlow. I trained it and evaluated it on the [moons dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

1. Defined the graph within a logistic_regression() function that can be reused easily.
2. Saved checkpoints using a Saver at regular intervals during training.
3. Implemented restoration to the last checkpoint upon startup if the training was interrupted.
4. Tweaked the hyperparameters such as the learning rate and the mini-batch size to get a better accuracy.


## Multi-Layer Perceptron (MLP)

I trained a deep Multi-layer perception (MLP) on the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist)

1. Loaded the MNIST dataset
2. Normalized the pixel values to the range [0, 1] and one-hot encoded the target labels (y_train and y_test).
3. Implemented the MLP model with the shown architecture:
   * An input layer with 256 neurons and ReLU activation.
   * A hidden layer with 128 neurons and ReLU activation.
   * An output layer with 10 neurons (corresponding to the 10 digits) and softmax activation.
   * The model is compiled with the Adam optimizer and categorical cross-entropy loss.
4. Implemented restoration to the last checkpoint upon startup if the training was interrupted.
5. Calculated Precision, recall, and F1 score.
