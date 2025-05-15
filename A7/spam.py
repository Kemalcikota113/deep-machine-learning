# Modify only the file you wish to load.
# Each data file contains 3 keys: X, y, msg
import numpy as np

data = np.load(file='2048d_sms_spam_albert-xlarge-v2.npz')
# data = np.load(file='300d_sms_spam_fasttext_pca.npz')

X: np.ndarray; Y: np.ndarray
X, Y, msg = data.get('X'), data.get('y'), data.get('msg')
X.shape, Y.shape, len(msg), msg[2], Y[2]

import torch
from torch import Tensor

print(torch.__version__) # Just to check the version and that torch is loaded
print("CUDA available:", torch.cuda.is_available())


def model(w: Tensor, x: Tensor) -> Tensor:
    """
    Takes the coefficients w, the observations, computes the polynomial and
    returns probabilities.
    w: Tensor of shape (..., D+1) containing [w0, w1, ..., wD]
    x: Tensor of shape (N, D) containing D-dimensional inputs for N samples

    Returns:
        Tensor of shape (N,) with predicted probabilities in [0,1]
    """
    # Number of features
    D = x.shape[1]
    # Ensure w has D+1 coefficients
    assert w.shape[-1] == D + 1, f"Expected w[..., D+1], got {w.shape[-1]}"

    # Compute the polynomial z = w0 + w1*x1 + w2*(x2^2) + ... + wD*(xD^D)
    # Broadcast w terms against x dimensions
    z = w[..., 0]  # bias term, shape broadcastable to (N,)
    for j in range(1, D + 1):
        # x[:, j-1] raised to the j-th power, shape (N,)
        term = x[:, j-1] ** j
        # w[..., j] has shape like (...,)
        z = z + w[..., j] * term
    # Apply sigmoid to get probabilities
    return torch.sigmoid(z)


# Sanity-check, we should get 4 outputs if the model is correctly vectorized!
temp = model(w=torch.rand(size=(1,6)), x=torch.rand(size=(4,5)))
temp

def log_lik(y_true: Tensor, y_hat: Tensor) -> Tensor:
    """
    For one or more observations, where y_true is the true label (0 or 1)
    and y_hat contains predicted probabilities [0,1], computes the (log)
    likelihood summed over all observations.

    Returns a single-element tensor.
    """
    # Avoid log(0) by clamping y_hat within (eps, 1-eps)
    eps = 1e-8
    p = y_hat.clamp(min=eps, max=1.0 - eps)
    # Log-likelihood: sum_i [y_i*log(p_i) + (1-y_i)*log(1-p_i)]
    ll = (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p)).sum()
    return ll
#tensor([0.7839, 0.7999, 0.7693, 0.8770])

# Sanity-check, should be a single element here:
log_lik(y_true=torch.tensor(data=[1,0,1,0], dtype=torch.float), y_hat=temp)

# output that was provided (double check):
# tensor(-3.9837)