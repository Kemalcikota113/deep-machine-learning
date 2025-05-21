import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.load(file='2048d_sms_spam_albert-xlarge-v2.npz')
# data = np.load(file='300d_sms_spam_fasttext_pca.npz')

embeddings = data['X']
labels = data['y']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# PCA: at least 30% variance, at most 15 components
pca = PCA(n_components=min(15, X_train.shape[1]))
pca.fit(X_train)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = int(np.searchsorted(cumvar, 0.30) + 1)
print(f"Using {n_components} PCA components capturing {cumvar[n_components-1]:.3f} variance")

pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

scaler2 = StandardScaler() # scale and standardize again after PCA
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)


# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32)

# Mini-batch loader
BATCH_SIZE = 64
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --- MODEL AND HELPER FUNCTIONS ---

POLY_DEGREE = n_components  # use PCA components as degree

def model(w: Tensor, x: Tensor) -> Tensor:
    """
    Polynomial logistic model: z = w0 + Σ_{j=1}^D w_j * x_j^j, then sigmoid.
    - w: (..., D+1)
    - x: (N, D)
    Returns: Tensor of shape (N,) probabilities.
    """
    D = x.shape[1]
    assert w.shape[-1] == D + 1
    z = w[..., 0]
    for j in range(1, D+1):
        # More aggressive scaling of polynomial terms
        z = z + w[..., j] * (x[:, j-1] ** j) / (j * 100)
    return torch.sigmoid(z)


def log_lik(y_true: Tensor, y_hat: Tensor) -> Tensor:
    """Sum log-likelihood for Bernoulli outcomes."""
    eps = 1e-8
    p = y_hat.clamp(eps, 1 - eps)
    # Use log-sum-exp trick for numerical stability
    log_p = torch.log(p)
    log_1_p = torch.log(1 - p)
    # Add small epsilon to prevent -inf
    return (y_true * log_p + (1-y_true) * log_1_p + eps).sum()


# --------
# CELL 2
# --------

NUM_VARIATIONAL_SETS = 50

def ELBO_expected_data_likelihood(y_true: Tensor, x: Tensor, W: Tensor) -> Tensor:
    """A function to calculate the first term of the ELBO.    Monte Carlo estimate of E_{q}[log p(y|x,w)].
    - W: shape (S, D+1)
    """
    device = y_true.device
    ll_sum = torch.zeros((), device=device)
    S = W.shape[0]
    for s in range(S):
        w_s = W[s]
        y_hat = model(w_s, x)
        # Add small epsilon to prevent log(0)
        y_hat = y_hat.clamp(min=1e-8, max=1-1e-8)
        ll = log_lik(y_true, y_hat)
        
        # Debug prints for first sample
        #if s == 0:
            #print(f"y_hat range: [{y_hat.min():.4f}, {y_hat.max():.4f}]")
            #print(f"log_lik: {ll:.4f}")
        
        ll_sum = ll_sum + ll
    
    return ll_sum / float(S)


def ELBO_KL_divergence_analytical(mu: Tensor, sigma: Tensor) -> Tensor:
    """The analytical version of the KL divergence.    KL[q||p] for q = N(mu, diag(sigma^2)), p = N(0,I).
    """
    # Add small epsilon to prevent log(0) and ensure numerical stability
    sigma_sq = sigma**2 + 1e-6
    # Clip mu to prevent extreme values
    mu_clipped = torch.clamp(mu, min=-10.0, max=10.0)
    # Ensure sigma_sq is not too small or too large
    sigma_sq = torch.clamp(sigma_sq, min=1e-6, max=1e6)
    
    kl = 0.5 * torch.sum(mu_clipped**2 + sigma_sq - 1 - torch.log(sigma_sq))
    
    # Debug prints
    #print("Debug ELBO_KL_divergence_analytical:")
    #print(f"mu range: [{mu.min():.4f}, {mu.max():.4f}]")
    #print(f"sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    #print(f"sigma_sq range: [{sigma_sq.min():.4f}, {sigma_sq.max():.4f}]")
    #print(f"kl: {kl:.4f}")
    
    return kl


def ELBO_KL_divergence_Monte_Carlo(W: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """
    MC estimate of KL divergence via samples W ~ q.
    W shape: (S, D+1)
    """
    S, Dp = W.shape
    # Add small epsilon to prevent division by zero
    sigma_safe = sigma + 1e-8
    
    # Debug prints if NaN
    #if torch.isnan(W).any() or torch.isnan(mu).any() or torch.isnan(sigma).any():
        #print("Debug ELBO_KL_divergence_Monte_Carlo:")
        #print(f"W range: [{W.min():.4f}, {W.max():.4f}]")
        #print(f"mu range: [{mu.min():.4f}, {mu.max():.4f}]")
        #print(f"sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    
    # log q and log p
    log_q = -0.5 * (((W - mu) / sigma_safe) ** 2 + torch.log(2 * torch.pi * sigma_safe**2))
    log_p = -0.5 * (W**2 + torch.log(2 * torch.pi))
    
    # Debug prints if NaN
    #if torch.isnan(log_q).any() or torch.isnan(log_p).any():
        #print(f"log_q range: [{log_q.min():.4f}, {log_q.max():.4f}]")
        #print(f"log_p range: [{log_p.min():.4f}, {log_p.max():.4f}]")
    
    # sum over dims, mean over samples
    return torch.mean(torch.sum(log_q - log_p, dim=1))

# --------
# CELL 3
# --------

from typing import Literal

NUM_VARIATIONAL_SETS = 50
KL_DIV_TYPE = Literal['analytical', 'montecarlo']


def ELBO(use_mu: Tensor, use_sigma: Tensor, y_true: Tensor, obs: Tensor, variational_params_noise: Tensor, kl: KL_DIV_TYPE='analytical', return_exp_data_lik: bool=False, return_kl_div: bool=False) -> Tensor|tuple[Tensor, ...]:
    """
    Convenience function that uses the current variational parameters,
    applies the reparameterization trick, and computes the complete
    ELBO. The result of this function shall be maximized.
    """
    # Reparameterization: W = mu + sigma * eps
    W = use_mu.unsqueeze(0) + use_sigma.unsqueeze(0) * variational_params_noise   # (S, D+1)
    exp_lik = ELBO_expected_data_likelihood(y_true, obs, W)
    if kl == 'analytical':
        kl_div = ELBO_KL_divergence_analytical(use_mu, use_sigma)
    else:
        kl_div = ELBO_KL_divergence_Monte_Carlo(W, use_mu, use_sigma)
    elbo = exp_lik - kl_div
    outputs = (elbo,)
    if return_exp_data_lik: outputs += (exp_lik,)
    if return_kl_div:      outputs += (kl_div,)
    return outputs[0] if len(outputs)==1 else outputs


# TODO: Define/create the gradient of the ELBO function.
#ELBO_grad = ...

# --------
# CELL 4
# --------

# TODO: Implement as however you require this!
# TODO: Make sure to use the finally found optimal parameters once you submit your solution.

D_plus1 = X_train_t.shape[1] + 1
mu_param       = torch.zeros(D_plus1, requires_grad=True)
log_sigma_param = torch.zeros(D_plus1, requires_grad=True)

optimizer = torch.optim.Adam([mu_param, log_sigma_param], lr=1e-3)


EPOCHS = 50
LEARNING_RATE = 0.0001
USE_KL_TYPE: KL_DIV_TYPE = 'analytical'
elbo_history = []

for epoch in range(EPOCHS):
    epoch_elbo = 0.0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        sigma = torch.exp(log_sigma_param)
        eps = torch.randn(NUM_VARIATIONAL_SETS, D_plus1)
        elbo = ELBO(mu_param, sigma, yb, Xb, eps, kl=USE_KL_TYPE)
        loss = -elbo
        loss.backward()
        optimizer.step()
        epoch_elbo += elbo.item()
    elbo_history.append(epoch_elbo / len(train_loader))
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} ELBO: {elbo_history[-1]:.4f}")

# Final variational parameters
final_mu = mu_param.detach()
final_sigma = torch.exp(log_sigma_param).detach()
print("Final mu:", final_mu)
print("Final sigma:", final_sigma)

# ------------------------
# 6. Evaluation & Plotting
# ------------------------
# Plot training ELBO
plt.plot(elbo_history)
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.title('ELBO over Training')
plt.show()

# ELBO on test data
eps_test = torch.randn(NUM_VARIATIONAL_SETS, D_plus1)
test_elbo = ELBO(mu_param, torch.exp(log_sigma_param), y_test_t, X_test_t, eps_test, kl=USE_KL_TYPE)
print(f"Test set ELBO: {test_elbo:.4f}")
# per sample ELBO

# --------------------
# cell 5
# --------------------

test_elbo_per_sample = test_elbo.item() / len(y_test_t)
print(f"Test set ELBO per sample: {test_elbo_per_sample:.4f}")

# ----------------- 
# ADVANCED MODEL EVALUATION
# -----------------

# 4. Posterior Weight Visualization
import matplotlib.pyplot as plt

final_mu    = mu_param.detach()
final_sigma = torch.exp(log_sigma_param).clamp(min=1e-3).detach()
print("Final mu:", final_mu)
print("Final sigma:", final_sigma)

# Move variational parameters to CPU numpy
weights_mu = final_mu.cpu().numpy()
weights_sigma = final_sigma.cpu().numpy()
param_names = [f"w{i}" for i in range(len(weights_mu))]

# Plot posterior means with ±1 std-error bars
plt.figure(figsize=(10, 4))
plt.bar(param_names, weights_mu, yerr=weights_sigma, capsize=4)
plt.axhline(0, color='gray', linewidth=0.8)
plt.xlabel('Parameter')
plt.ylabel('Posterior mean ± SD')
plt.title('Posterior Weight Estimates with Uncertainty')
plt.tight_layout()
plt.show()

# Short explanation
#print("
#Posterior Weight Visualization Explanation:
#")
#print("We plot each coefficient's posterior mean (μ) with an error bar representing one standard deviation (σ).
#")
#print("Parameters whose σ remains near 1 indicate that the approximate posterior remains at the prior;
#")
#print("parameters with σ much smaller than 1 show that the model is confident in that weight's deviation from zero.")