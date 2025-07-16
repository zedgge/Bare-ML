This project is a fully self-contained neural network implementation in Python, written entirely from scratch without using any external libraries — not even NumPy.

The model is designed to classify handwritten digits from the MNIST dataset. It includes training, testing, a clean CLI flow, modern activation and optimization strategies, and is crafted for both educational purposes and practical performance.


Model Architecture:

Input Layer: 784 nodes (28x28 pixel images)

Hidden Layer: 128 ReLU neurons

Output Layer: 10 nodes (Softmax for classification 0–9)

Training Approach:

Supervised learning with cross-entropy loss

Only trains on samples it hasn’t mastered (based on repeated correctness)

Learning rate decay: 0.7 every 5 epochs

🔬 Code Structure Overview

dot(), matmul() — Core math operations

relu(), softmax() — Activations

log() — Manual approximation for natural log

NeuralNetwork — Forward + backward propagation

train_until_mastery() — Memory-efficient loop

test() — Evaluation on holdout set

main() — Loads data, trains, tests

📈 Performance

Training time: ~30–60 seconds for 1,000 samples

Accuracy: ~90–92% on test set with only 1,000 training examples (results were performed on a laptop)

Memory-efficient: Only keeps data and model in RAM

You can scale up by changing MAX_TRAIN and MAX_TEST in the script:

MAX_TRAIN = 60000
MAX_TEST = 10000

🛠️ Advanced Options

Want to extend the project?

Add model saving/loading via basic file I/O

Add a second hidden layer or dropout

Implement batch training

Add support for multi-core processing with multiprocessing

All without libraries like NumPy, TensorFlow, or PyTorch.

📚 Learning Goals

This project is excellent for anyone learning about:

Neural network internals

Manual gradient computation

Backpropagation without frameworks

How deep learning works under the hood

Pure Python computational graph logic

🤝 Contributing

Pull requests, suggestions, and forks are welcome!
You can:

Improve the math functions

Add model save/load support

Add visualization of accuracy over epochs

