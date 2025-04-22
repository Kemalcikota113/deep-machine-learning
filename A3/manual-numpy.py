import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─── 1) XOR DATA ────────────────────────────────────────────
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)
Y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)

# ─── 2) MATLAB‐exported parameters ─────────────────────────
w11 = np.array([ 0.41652033,  0.7749157 ,  0.83778775,  0.64122725,  0.3972386 ,
                -0.13516521, -0.60432655, -0.20290181, -0.7232665 , -0.35317   ,
                -0.8889785 , -0.2863993 ,  0.54030126,  0.37175888, -0.82367843,
                -0.33497408, -0.8567182 ,  0.5161414 , -0.25462216,  0.59564906,
                 0.80922335,  0.00252599,  0.3720131 , -0.1701207 ,  0.34805146,
                -0.38333094, -0.17244412,  0.7132528 , -0.14648864, -0.17876345,
                -0.6636416 , -0.39717668], dtype=np.float32)

w12 = np.array([ 0.41662493, -0.7749062 , -0.8376726 , -0.64101034,  0.40976927,
                -0.09223783,  0.60428804, -0.09886253,  0.7417394 , -0.08979583,
                 1.1194593 ,  0.13293397,  0.5400226 ,  0.37171456,  0.8246372 ,
                -0.43565178,  0.85677105,  0.51622313, -0.41895077, -0.5953253 ,
                -0.8091849 , -0.3766787 ,  0.28819063, -0.38122708,  0.34833792,
                 0.15245543, -0.25060123, -0.7132655 , -0.30406952, -0.3443008 ,
                 0.66343206, -0.03848529], dtype=np.float32)

b1 = np.array([-4.16518986e-01, -4.11640503e-05, -1.45169579e-05, -3.96516771e-06,
               -1.09622735e-04,  0.00000000e+00,  2.90098578e-05,  0.00000000e+00,
               -1.63983065e-03,  0.00000000e+00, -6.36966957e-04, -1.47912502e-01,
               -5.39916575e-01, -3.71752203e-01, -5.50682598e-04,  1.15736163e+00,
               -6.27825721e-05, -5.16140461e-01,  0.00000000e+00, -1.62737619e-04,
               -8.02413124e-05, -2.25486048e-02,  1.79730232e-05,  0.00000000e+00,
               -3.48306417e-01, -1.66664287e-01,  0.00000000e+00, -3.03934885e-05,
                0.00000000e+00,  0.00000000e+00,  6.44910688e-05,  0.00000000e+00],
              dtype=np.float32)

ws2 = np.array([-0.8178053 ,  1.0259871 ,  0.7672013 ,  0.9983522 ,  0.3300904 ,
                 0.1318875 ,  0.9221373 ,  0.20387506,  1.0057546 ,  0.2067241 ,
                 0.7479725 , -0.25578788, -0.83140665, -0.952559  ,  0.946334  ,
                -1.0112743 ,  0.5946371 , -1.0230583 , -0.04098088,  0.9588608 ,
                 0.7448052 , -0.10435582,  0.25728533,  0.30141222, -0.8235109 ,
                -0.21171094, -0.28320622,  1.0405817 ,  0.40911728,  0.37331975,
                 0.87603277,  0.19098908, -0.8885394], dtype=np.float32)

# ─── 3) WRAP INTO W1,b1,W2,b2 ─────────────────────────────────
# W1: shape (2,32)
W1 = np.stack([w11, w12], axis=0)          # (2 inputs → 32 hidden)
b1 = b1                                    # (32,)

# W2: shape (32,1), b2: (1,)
W2 = ws2[:32].reshape(32,1)
b2 = ws2[32:].reshape(1,)

# ─── 4) ACTIVATIONS & UTILITIES ─────────────────────────────
def relu(z):           return np.maximum(0, z)
def relu_deriv(z):     return (z > 0).astype(np.float32)
def sigmoid(z):        return 1/(1+np.exp(-z))
def sigmoid_deriv(z):  s = sigmoid(z); return s*(1-s)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1          # → (batch,32)
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2         # → (batch,1)
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def mse(A2, Y):
    return np.mean((Y - A2)**2)

# ─── 5) INITIAL LOSS & PREDICTION ────────────────────────────
_, _, _, A20 = forward(X, W1, b1, W2, b2)
print("Initial MSE:", mse(A20, Y))

def predict(X):
    return np.round(forward(X, W1, b1, W2, b2)[3]).squeeze()

print("Initial pred:", predict(X).reshape(-1,1).T)

# ─── 6) TRAINING LOOP ────────────────────────────────────────
lr, epochs = 1.5, 500
loss_hist = []

for _ in range(epochs):
    Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)

    # backward
    dZ2 = (A2 - Y) * sigmoid_deriv(Z2)      # (4×1)
    dW2 = A1.T @ dZ2                        # (32×4)@(4×1) → (32,1)
    db2 = dZ2.sum(axis=0)                   # (1,)

    dA1 = dZ2 @ W2.T                        # (4×1)@(1×32) → (4,32)
    dZ1 = dA1 * relu_deriv(Z1)              # (4×32)
    dW1 = X.T @ dZ1                         # (2×4)@(4×32) → (2,32)
    db1 = dZ1.sum(axis=0)                   # (32,)

    # updates
    W2 -= lr * dW2;  b2 -= lr * db2
    W1 -= lr * dW1;  b1 -= lr * db1

    loss_hist.append(mse(A2, Y))

# ─── 7) PLOT LOSS & ACCURACY ────────────────────────────────
plt.plot(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("NumPy Training Loss")
plt.grid(True)
plt.show()

preds = predict(X)
acc = np.mean(preds == Y.squeeze())
print("Final XOR preds:", preds)
print("Accuracy:", acc)

# ─── 8) SURFACE PLOT ─────────────────────────────────────────
grid = 50
xs = np.linspace(0,1,grid); ys = np.linspace(0,1,grid)
Xg,Yg = np.meshgrid(xs,ys)
pts = np.column_stack([Xg.ravel(), Yg.ravel()])
Zg  = forward(pts, W1, b1, W2, b2)[3].reshape(Xg.shape)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Zg)
ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel('f(x₁,x₂)')
ax.set_title('NumPy XOR Net Surface')
plt.show()
