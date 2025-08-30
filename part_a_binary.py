import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import argparse

#  python part_a_binary.py --dataset_root 'dataset_for_A2//dataset_for_A2//binary_dataset' --save_weights_path "weights_a"

np.random.seed(0)

class CustomImageDataset:
    def __init__(self, root_dir, csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L") #Convert image to greyscale
        label = row["class"]

        if self.transform:
            image = self.transform(image)

        return np.array(image), label

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

class DataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)

    def __iter__(self):
        self.start_idx = 0
        return self
    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

    def __next__(self):
        if self.start_idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.start_idx:end_idx]
        images = []
        labels = []

        for idx in batch_indices:
            image, label = self.dataset[idx]
            images.append(image)
            labels.append(label)

        self.start_idx = end_idx

        # Stack images and labels to create batch tensors
        batch_images = np.stack(images, axis=0)
        batch_labels = np.array(labels)

        return batch_images, batch_labels
            
def sigmoid(z):
    # z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    Z4 = np.dot(A3, W4) + b4
    A4 = sigmoid(Z4)

    return A1, A2, A3, A4  # A4 is the output

def binary_cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def backward_pass(X, Y, A1, A2, A3, A4, W1, W2, W3, W4):
    m = X.shape[0]

    # Output layer gradients
    dZ4 = (A4 - Y ) # Shape (n, 1)
    dW4 = np.dot(A3.T, dZ4)/m # Shape (128, 1)
    db4 = np.sum(dZ4, axis=0,keepdims=True)/m

    # Layer 3 gradients
    dA3 = np.dot(dZ4, W4.T)  # Shape (n, 128)
    dZ3 = dA3 * A3 * (1 - A3)  # Sigmoid derivative
    dW3 = np.dot(A2.T, dZ3)/m # Shape (265, 128)
    db3 = np.sum(dZ3, axis=0,keepdims=True)/m  # Shape (1, 128)

    # Layer 2 gradients
    dA2 = np.dot(dZ3, W3.T)  # Shape (n, 265)
    dZ2 = dA2 * A2 * (1 - A2)  # Sigmoid derivative
    dW2 = np.dot(A1.T, dZ2)/m  # Shape (512, 265)
    db2 = np.sum(dZ2, axis=0,keepdims=True)/m  # Shape (1, 265)

    # Layer 1 gradients
    dA1 = np.dot(dZ2, W2.T)  # Shape (n, 512)
    dZ1 = dA1 * A1 * (1 - A1)  # Sigmoid derivative
    dW1 = np.dot(X.T, dZ1)/m # Shape (625, 512)
    db1 = np.sum(dZ1, axis=0,keepdims=True)/m # Shape (1, 512)

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4

def init_params():
  np.random.seed(0)
  W1 = np.random.randn(625, 512) * np.sqrt(2 / (625))
  b1 = np.zeros((1, 512)) # row vector

  W2 = np.random.randn(512, 256) * np.sqrt(2 / (512))
  b2 = np.zeros((1, 256)) # row vector

  W3 = np.random.randn(256, 128) * np.sqrt(2 / (256))
  b3 = np.zeros((1, 128)) # row vector

  W4 = np.random.randn(128, 1) * np.sqrt(2 / (128))
  b4 = np.zeros((1, 1)) # row vector

  return W1,b1,W2,b2,W3,b3,W4,b4

def update_params(W1,b1,W2,b2,W3,b3,W4,b4,dW1,db1,dW2,db2,dW3,db3,dW4,db4,lr):
  W1 = W1 - lr*dW1
  b1 = b1 - lr*db1

  W2 = W2 - lr*dW2
  b2 = b2 - lr*db2

  W3 = W3 - lr*dW3
  b3 = b3 - lr*db3

  W4 = W4 - lr*dW4
  b4 = b4 - lr*db4

  return W1,b1,W2,b2,W3,b3,W4,b4

def train(W1,b1,W2,b2,W3,b3,W4,b4, X, Y, epochs, batch_size, lr):
  for j in range(epochs):
    output = None
    X_batch = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
    Y_batch = [Y[i:i + batch_size] for i in range(0, len(Y), batch_size)]


    for X_b,Y_b in zip(X_batch,Y_batch):
      A1, A2, A3, A4 = forward_pass(X_b,W1,b1,W2,b2,W3,b3,W4,b4)
      dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_pass(X_b, Y_b, A1, A2, A3, A4, W1, W2, W3, W4)
      W1,b1,W2,b2,W3,b3,W4,b4 = update_params(W1,b1,W2,b2,W3,b3,W4,b4,dW1,db1,dW2,db2,dW3,db3,dW4,db4,lr)

  return W1,b1,W2,b2,W3,b3,W4,b4

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

if __name__ == '__main__':
    # Root directory containing the 8 subfolders
    ## root_dir = "/home/kaustubh/scratch/COL774_A2_TA_files/dataset_for_assignment_binary/"
    root_dir = 'dataset_for_A2//dataset_for_A2//binary_dataset'
    ## mode = None #Set mode to 'train' for loading the train set for training. Set mode to 'val' for testing your model after training. 
    mode = 'train'


    parser = argparse.ArgumentParser(description="Train ANN for binary classification")
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to the train dataset root folder")
    parser.add_argument('--save_weights_path', type=str, required=True, help="Path to save the model weights")
    args = parser.parse_args()

    root_dir=args.dataset_root

    if mode == 'train': # Set mode to train when using the dataloader for training the model.
        csv = os.path.join(root_dir, "train.csv")

    elif mode == 'val':
        csv = os.path.join(root_dir, "val.csv")


    # Create the custom dataset
    dataset = CustomImageDataset(root_dir=root_dir, csv = csv, transform=numpy_transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=1)

    image_list = []
    label_list = []

    for images, labels in dataloader:
        image_list.append(images.flatten()) 
        label_list.append(labels) 

    X = pd.DataFrame(image_list).to_numpy() 
    Y = pd.DataFrame(label_list, columns=['Label']).to_numpy() 

    
    epochs = 15
    lr = 0.001
    batch_size = 256


    W1,b1,W2,b2,W3,b3,W4,b4 = init_params()

    W1,b1,W2,b2,W3,b3,W4,b4 = train(W1,b1,W2,b2,W3,b3,W4,b4, X, Y, epochs, batch_size, lr)
    
        
    weights = {
        'fc1': W1,
        'fc2': W2,
        'fc3': W3,
        'fc4': W4
    }

    biases = {
        'fc1': b1.flatten(),
        'fc2': b2.flatten(),
        'fc3': b3.flatten(),
        'fc4': b4.flatten()
    }

    weights_path = os.path.join(args.save_weights_path, "weights.pkl")

    save_weights_biases(weights, biases, weights_path)



