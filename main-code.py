# Math Utilities
def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def matmul(vec, mat):
    return [dot(vec, col) for col in zip(*mat)]

def relu(x):
    return x if x > 0 else 0

def relu_deriv(x):
    return 1 if x > 0 else 0

def softmax(x):
    max_x = max(x)
    exps = [pow(2.718281828459045, i - max_x) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]

def cross_entropy(y_pred, y_true):
    return -sum(y_true[i] * log(y_pred[i] + 1e-9) for i in range(len(y_true)))

def argmax(vec):
    max_i = 0
    for i in range(1, len(vec)):
        if vec[i] > vec[max_i]:
            max_i = i
    return max_i

def one_hot(label, size=10):
    vec = [0.0] * size
    vec[label] = 1.0
    return vec

def load_csv(path, max_rows):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(1, min(len(lines), max_rows + 1)):
        parts = lines[i].strip().split(',')
        label = int(parts[0])
        pixels = [float(p) / 255.0 for p in parts[1:]]
        data.append((pixels, one_hot(label)))
    return data

def log(x):
    n = 100
    return n * ((x ** (1 / n)) - 1) if x > 0 else -999

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = [[((i * j % 7) - 3) / 10.0 for j in range(hidden_size)] for i in range(input_size)]
        self.b1 = [0.01] * hidden_size
        self.w2 = [[((i * j % 5) - 2) / 10.0 for j in range(output_size)] for i in range(hidden_size)]
        self.b2 = [0.01] * output_size

    def forward(self, x):
        self.z1 = [dot(x, col) + self.b1[i] for i, col in enumerate(zip(*self.w1))]
        self.a1 = [relu(z) for z in self.z1]
        self.z2 = [dot(self.a1, col) + self.b2[i] for i, col in enumerate(zip(*self.w2))]
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, x, y, lr):
        dz2 = [self.a2[i] - y[i] for i in range(len(y))]
        dw2 = [[self.a1[i] * dz2[j] for j in range(len(dz2))] for i in range(len(self.a1))]
        db2 = dz2[:]

        da1 = [sum(self.w2[i][j] * dz2[j] for j in range(len(dz2))) for i in range(len(self.a1))]
        dz1 = [da1[i] * relu_deriv(self.z1[i]) for i in range(len(da1))]
        dw1 = [[x[i] * dz1[j] for j in range(len(dz1))] for i in range(len(x))]
        db1 = dz1[:]

        for i in range(len(self.w2)):
            for j in range(len(self.w2[0])):
                self.w2[i][j] -= lr * dw2[i][j]
        for i in range(len(self.b2)):
            self.b2[i] -= lr * db2[i]

        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])):
                self.w1[i][j] -= lr * dw1[i][j]
        for i in range(len(self.b1)):
            self.b1[i] -= lr * db1[i]
            
def train_until_mastery(net, data, max_epochs, lr, mastery_threshold=3):
    memory = [0] * len(data)
    decay = 0.7
    for epoch in range(1, max_epochs + 1):
        correct = 0
        total_loss = 0.0
        used = 0

        for i in range(len(data)):
            if memory[i] >= mastery_threshold:
                continue
            x, y = data[i]
            out = net.forward(x)
            pred = argmax(out)
            actual = argmax(y)

            if pred == actual:
                memory[i] += 1
                correct += 1
            else:
                memory[i] = 0
                net.backward(x, y, lr)

            total_loss += cross_entropy(out, y)
            used += 1

        acc = correct / max(1, used) * 100
        avg_loss = total_loss / max(1, used)
        print(f"Epoch {epoch} | Active: {used} | Accuracy: {round(acc, 2)}% | Avg Loss: {round(avg_loss, 4)}")

        if all(m >= mastery_threshold for m in memory):
            print("[✓] Mastered all training samples.")
            break

        if epoch % 5 == 0:
            lr *= decay

    return memory


def test(net, data):
    correct = 0
    for x, y in data:
        out = net.forward(x)
        if argmax(out) == argmax(y):
            correct += 1
    return correct / len(data) * 100

def main():
    TRAIN_PATH = "mnist_train.csv"  #recomended to use direct file path for reliability
    TEST_PATH = "mnist_test.csv"  #recomended to use direct file path for reliability 
    MAX_TRAIN = 1000
    MAX_TEST = 200

    print("[+] Loading data...")
    train_data = load_csv(TRAIN_PATH, MAX_TRAIN)
    test_data = load_csv(TEST_PATH, MAX_TEST)

    print("[+] Initializing network...")
    net = NeuralNetwork(784, 128, 10)

    print("[+] Training...")
    train_until_mastery(net, train_data, max_epochs=50, lr=0.15)

    print("[+] Testing...")
    acc = test(net, test_data)
    print("Test Accuracy:", round(acc, 2), "%")

main()
