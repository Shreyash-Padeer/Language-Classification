import numpy as np
import pandas as pd
from part_d_trainloader import TrainImageDataset
from part_d_trainloader import TrainDataLoader 
from part_d_testloader import TestImageDataset
from part_d_testloader import TestDataLoader
import argparse
import os
import time
import pickle

# python part_d_best.py --dataset_root "dataset_for_A2\dataset_for_A2\multi_dataset" --test_dataset_root "dataset_for_A2\dataset_for_A2\multi_dataset" --save_weights_path "weights_d" --save_predictions_path "predictions_d"

# Transformations using NumPy
def resize(image, size):
    # return np.array(Image.fromarray(image).resize(size))
    return np.array(image.resize(size))

def to_tensor(image):
    return image.astype(np.float32) / 255.0

def numpy_transform(image, size=(25, 25)):
    image = resize(image, size)
    image = to_tensor(image)
    image = image.flatten()
    return image

def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def softmax(X):
    X_exp = np.exp(X - np.max(X, axis=1, keepdims=True))
    softmax_X = X_exp / np.sum(X_exp, axis=1, keepdims=True)
    return softmax_X

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def init_params():
  np.random.seed(0)

  W1 = np.random.randn(625, 512) * np.sqrt(2 / (625))
  b1 = np.zeros((1, 512)) # row vector

  W2 = np.random.randn(512, 256) * np.sqrt(2 / (512))
  b2 = np.zeros((1, 256)) # row vector

  W3 = np.random.randn(256, 128) * np.sqrt(2 / (256))
  b3 = np.zeros((1, 128)) # row vector

  W4 = np.random.randn(128, 32) * np.sqrt(2 / (128))
  b4 = np.zeros((1, 32)) # row vector

  W5 = np.random.randn(32, 8) * np.sqrt(2 / (32))
  b5 = np.zeros((1, 8)) # row vector

  return W1,b1,W2,b2,W3,b3,W4,b4,W5,b5

def forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    Z4 = np.dot(A3, W4) + b4
    A4 = sigmoid(Z4)

    Z5 = np.dot(A4, W5) + b5
    A5 = softmax(Z5)

    return A1, A2, A3, A4, A5  # A4 is the output

def backward_pass(X, Y, A1, A2, A3, A4, A5, W1, W2, W3, W4, W5):
    m = X.shape[0]

    # Output layer gradients
    dZ5 = (A5 - Y)
    dW5 = np.dot(A4.T, dZ5)/m
    db5 = np.sum(dZ5, axis=0,keepdims=True)/m

    # Layer 4 gradients
    dA4 = np.dot(dZ5, W5.T)
    dZ4 = dA4 * A4 * (1 - A4)
    dW4 = np.dot(A3.T, dZ4)/m
    db4 = np.sum(dZ4, axis=0,keepdims=True)/m

    # Layer 3 gradients
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * A3 * (1 - A3)
    dW3 = np.dot(A2.T, dZ3)/m
    db3 = np.sum(dZ3, axis=0,keepdims=True)/m

    # Layer 2 gradients
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * A2 * (1 - A2)
    dW2 = np.dot(A1.T, dZ2)/m
    db2 = np.sum(dZ2, axis=0,keepdims=True)/m

    # Layer 1 gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1)/m
    db1 = np.sum(dZ1, axis=0,keepdims=True)/m

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5

def update_params(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,lr):
  W1 = W1 - lr*dW1
  b1 = b1 - lr*db1

  W2 = W2 - lr*dW2
  b2 = b2 - lr*db2

  W3 = W3 - lr*dW3
  b3 = b3 - lr*db3

  W4 = W4 - lr*dW4
  b4 = b4 - lr*db4

  W5 = W5 - lr*dW5
  b5 = b5 - lr*db5

  return W1,b1,W2,b2,W3,b3,W4,b4,W5,b5

def train(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5, X, Y, epochs, method, batch_size, lr, beta,beta1,beta2,epsilon):
  gW1,gb1,gW2,gb2,gW3,gb3,gW4,gb4,gW5,gb5,loss,least_loss,time1 = 0,0,0,0,0,0,0,0,0,0,np.inf,np.inf,0
  best_wb = []

  if method == 'adam':
        mW1, mb1, mW2, mb2, mW3, mb3, mW4, mb4, mW5, mb5 = [np.zeros_like(w) for w in [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5]]
        vW1, vb1, vW2, vb2, vW3, vb3, vW4, vb4, vW5, vb5 = [np.zeros_like(w) for w in [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5]]

  itr = 1
  st = time.time() / 60
  for epoch in range(epochs):

    ct = time.time() / 60
    time1 = ct - st

    if time1 > 15:
      print("Time limit exceeded!")
      break


    X_batch = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    Y_batch = [Y[i:i + batch_size] for i in range(0, len(Y), batch_size)]

    for X_b,Y_b in zip(X_batch,Y_batch):
      A1, A2, A3, A4, A5 = forward_pass(X_b,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5)
      dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5 = backward_pass(X_b, Y_b, A1, A2, A3, A4, A5, W1, W2, W3, W4, W5)

      if method=='vanilla':
        W1,b1,W2,b2,W3,b3,W4,b4,W5,b5 = update_params(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,lr)

      if method == 'adam':
        gW1,gb1,gW2,gb2,gW3,gb3,gW4,gb4,gW5,gb5 = adam(beta1,beta2,epsilon,itr,dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,mW1,mb1,mW2,mb2,mW3,mb3,mW4,mb4,mW5,mb5,vW1,vb1,vW2,vb2,vW3,vb3,vW4,vb4,vW5,vb5)
        W1,b1,W2,b2,W3,b3,W4,b4,W5,b5 = update_params(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,gW1,gb1,gW2,gb2,gW3,gb3,gW4,gb4,gW5,gb5,lr)

      itr = itr + 1
    loss = calc_loss(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,X,Y)
    if loss < least_loss:
      least_loss = loss
      best_wb = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5]
      print("Least loss found! Weights saved!")

    print(f"Epoch = {epoch+1}, Least_Loss = {least_loss}, Time = {time1}")


  return best_wb[0],best_wb[1],best_wb[2],best_wb[3],best_wb[4],best_wb[5],best_wb[6],best_wb[7],best_wb[8],best_wb[9]

def calc_loss(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,X,Y):
  A1,A2,A3,A4,A5 = forward_pass(X,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5)

  loss = -np.sum(Y * np.log(A5 + 1e-10)) / A5.shape[0]

  return loss

def one_hot(Y):
    Y = Y.astype(np.int64)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

# Implementation of the Adam optimizer function
def adam(beta1, beta2, epsilon, t, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, mW1, mb1, mW2, mb2, mW3, mb3, mW4, mb4, mW5, mb5, vW1, vb1, vW2, vb2, vW3, vb3, vW4, vb4, vW5, vb5):
    # Update biased first moment estimate
    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    mb1 = beta1 * mb1 + (1 - beta1) * db1
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    mb2 = beta1 * mb2 + (1 - beta1) * db2
    mW3 = beta1 * mW3 + (1 - beta1) * dW3
    mb3 = beta1 * mb3 + (1 - beta1) * db3
    mW4 = beta1 * mW4 + (1 - beta1) * dW4
    mb4 = beta1 * mb4 + (1 - beta1) * db4
    mW5 = beta1 * mW5 + (1 - beta1) * dW5
    mb5 = beta1 * mb5 + (1 - beta1) * db5

    # Update biased second raw moment estimate
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1**2)
    vb1 = beta2 * vb1 + (1 - beta2) * (db1**2)
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2**2)
    vb2 = beta2 * vb2 + (1 - beta2) * (db2**2)
    vW3 = beta2 * vW3 + (1 - beta2) * (dW3**2)
    vb3 = beta2 * vb3 + (1 - beta2) * (db3**2)
    vW4 = beta2 * vW4 + (1 - beta2) * (dW4**2)
    vb4 = beta2 * vb4 + (1 - beta2) * (db4**2)
    vW5 = beta2 * vW5 + (1 - beta2) * (dW5**2)
    vb5 = beta2 * vb5 + (1 - beta2) * (db5**2)

    # Compute bias-corrected first moment estimate
    mW1_corr = mW1 / (1 - beta1**t)
    mb1_corr = mb1 / (1 - beta1**t)
    mW2_corr = mW2 / (1 - beta1**t)
    mb2_corr = mb2 / (1 - beta1**t)
    mW3_corr = mW3 / (1 - beta1**t)
    mb3_corr = mb3 / (1 - beta1**t)
    mW4_corr = mW4 / (1 - beta1**t)
    mb4_corr = mb4 / (1 - beta1**t)
    mW5_corr = mW5 / (1 - beta1**t)
    mb5_corr = mb5 / (1 - beta1**t)

    # Compute bias-corrected second raw moment estimate
    vW1_corr = vW1 / (1 - beta2**t)
    vb1_corr = vb1 / (1 - beta2**t)
    vW2_corr = vW2 / (1 - beta2**t)
    vb2_corr = vb2 / (1 - beta2**t)
    vW3_corr = vW3 / (1 - beta2**t)
    vb3_corr = vb3 / (1 - beta2**t)
    vW4_corr = vW4 / (1 - beta2**t)
    vb4_corr = vb4 / (1 - beta2**t)
    vW5_corr = vW5 / (1 - beta2**t)
    vb5_corr = vb5 / (1 - beta2**t)

    # Compute Adam updates
    gW1 = mW1_corr / (np.sqrt(vW1_corr) + epsilon)
    gb1 = mb1_corr / (np.sqrt(vb1_corr) + epsilon)
    gW2 = mW2_corr / (np.sqrt(vW2_corr) + epsilon)
    gb2 = mb2_corr / (np.sqrt(vb2_corr) + epsilon)
    gW3 = mW3_corr / (np.sqrt(vW3_corr) + epsilon)
    gb3 = mb3_corr / (np.sqrt(vb3_corr) + epsilon)
    gW4 = mW4_corr / (np.sqrt(vW4_corr) + epsilon)
    gb4 = mb4_corr / (np.sqrt(vb4_corr) + epsilon)
    gW5 = mW5_corr / (np.sqrt(vW5_corr) + epsilon)
    gb5 = mb5_corr / (np.sqrt(vb5_corr) + epsilon)

    return gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4, gW5, gb5

def get_predictions(A):
    return np.argmax(A, 0)

def save_weights_biases(weights, biases, filename="weights.pkl"):
    model_params = {
        'weights': {},
        'bias': {}
    }

    for i in range(1, len(weights) + 1):
        layer_key = f'fc{i}'
        model_params['weights'][layer_key] = weights[layer_key]
        model_params['bias'][layer_key] = biases[layer_key]

    # Save the model parameters as a pickle file
    with open(filename, 'wb') as file:
        pickle.dump(model_params, file)

def save_predictions(predictions, filename="predictions.pkl"):
    # Create a dictionary of dictionaries

    # Save the model parameters as a pickle file
    with open(filename, 'wb') as file:
        pickle.dump(predictions, file)

parser = argparse.ArgumentParser(description="Train ANN for binary classification")
parser.add_argument('--dataset_root', type=str, required=True, help="Path to the train dataset root folder")
parser.add_argument('--test_dataset_root', type=str, required=True, help="Path to the test dataset root folder")
parser.add_argument('--save_weights_path', type=str, required=True, help="Path to save the model weights")
parser.add_argument('--save_predictions_path', type=str, required=True, help="Path to save the model predictions")
args = parser.parse_args()

# root_dir=args.dataset_root
# csv = os.path.join(root_dir, "train.csv")

# # Create the custom dataset
# dataset = TrainImageDataset(root_dir=root_dir, csv = csv, transform=numpy_transform)  #Remember to import "numpy_transforms" functions.
# # Create the DataLoader
# dataloader = TrainDataLoader(dataset, batch_size=32)

# image_list = []
# label_list = []

# for images, labels in dataloader:
#     image_list.append(images.flatten())  # Flatten the image array
#     label_list.append(labels)  # Store the label


# # Convert lists to DataFrames
# X = pd.DataFrame(image_list)  # X contains flattened images
# Y = pd.DataFrame(label_list, columns=['Label'])

root_dir=args.dataset_root
csv = os.path.join(root_dir, "train.csv")

# Create the custom dataset
dataset = TrainImageDataset(root_dir=root_dir, csv = csv, transform=numpy_transform)  #Remember to import "numpy_transforms" functions.

# Create the DataLoader
dataloader = TrainDataLoader(dataset, batch_size=1)

image_list = []
label_list = []

for images, labels in dataloader:
    image_list.append(images.flatten())  # Flatten the image array
    label_list.append(labels)  # Store the label


# Convert lists to DataFrames
X = pd.DataFrame(image_list)  # X contains flattened images
Y = pd.DataFrame(label_list, columns=['Label'])


# Test DATA

root_dir_val=args.test_dataset_root
csv_val = os.path.join(root_dir, "val.csv")

# Create the custom dataset
dataset_val = TestImageDataset(root_dir=root_dir_val, csv = csv_val, transform=numpy_transform)  #Remember to import "numpy_transforms" functions.
# Create the DataLoader
dataloader_val = TestDataLoader(dataset_val, batch_size=1)

image_list_val = []
# label_list_val = []

for images in dataloader_val:
    image_list_val.append(images.flatten())  # Flatten the image array
    # label_list_val.append(labels)  # Store the label


# Convert lists to DataFrames
X = pd.DataFrame(image_list)  # X contains flattened images
Y = pd.DataFrame(label_list, columns=['Label'])


X = X.to_numpy()
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std
Y = Y.to_numpy().ravel()
Y_ohe = one_hot(Y)

# Convert lists to DataFrames
X_test = pd.DataFrame(image_list_val)  # X contains flattened images
# Y_test = pd.DataFrame(label_list_val, columns=['Label'])

X_test = X_test.to_numpy()
X_test = (X_test - X_mean) / X_std
# Y_test = Y_test.to_numpy().ravel()


epochs = 35
lr = 0.0008
batch_size = 32
method = 'adam'
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

W1,b1,W2,b2,W3,b3,W4,b4,W5,b5 = init_params()

W1,b1,W2,b2,W3,b3,W4,b4,W5,b5 = train(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5, X, Y_ohe, epochs, method, batch_size, lr, beta,beta1,beta2,epsilon)

A1,A2,A3,A4,A5 = forward_pass(X_test,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5)

weights = {
    'fc1': W1,
    'fc2': W2,
    'fc3': W3,
    'fc4': W4,
    'fc5': W5
}

biases = {
    'fc1': b1.flatten(),
    'fc2': b2.flatten(),
    'fc3': b3.flatten(),
    'fc4': b4.flatten(),
    'fc5': b5.flatten()
}

weights_path = args.save_weights_path
save_weights_biases(weights, biases, weights_path)

predictions = get_predictions(A5.T)

predictions_path = args.save_predictions_path
save_predictions(predictions, predictions_path)

