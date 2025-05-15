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

# Add more aggressive normalization to ensure values are in a very reasonable range
X_train = np.clip(X_train, -3, 3)
X_test = np.clip(X_test, -3, 3)

# PCA: at least 30% variance, at most 15 components
pca = PCA(n_components=min(15, X_train.shape[1]))
pca.fit(X_train)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = int(np.searchsorted(cumvar, 0.30) + 1)
print(f"Using {n_components} PCA components capturing {cumvar[n_components-1]:.3f} variance")

pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

# Re-standardize after PCA
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

# Optionally clip again
X_train = np.clip(X_train, -3, 3)
X_test = np.clip(X_test, -3, 3)

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


NUM_VARIATIONAL_SETS = 50

def ELBO_expected_data_likelihood(y_true: Tensor, x: Tensor, W: Tensor) -> Tensor:
    """A function to calculate the first term of the ELBO.    Monte Carlo estimate of E_{q}[log p(y|x,w)].
    - W: shape (S, D+1)
    """
    device = y_true.device
    ll_sum = torch.zeros((), device=device)
    S = W.shape[0]
    
    # Debug prints for first call
    print("Debug ELBO_expected_data_likelihood:")
    print(f"W range: [{W.min():.4f}, {W.max():.4f}]")
    print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"y_true range: [{y_true.min():.4f}, {y_true.max():.4f}]")
    
    for s in range(S):
        w_s = W[s]
        y_hat = model(w_s, x)
        # Add small epsilon to prevent log(0)
        y_hat = y_hat.clamp(min=1e-8, max=1-1e-8)
        ll = log_lik(y_true, y_hat)
        
        # Debug prints for first sample
        if s == 0:
            print(f"y_hat range: [{y_hat.min():.4f}, {y_hat.max():.4f}]")
            print(f"log_lik: {ll:.4f}")
        
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
    print("Debug ELBO_KL_divergence_analytical:")
    print(f"mu range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"sigma_sq range: [{sigma_sq.min():.4f}, {sigma_sq.max():.4f}]")
    print(f"kl: {kl:.4f}")
    
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
    if torch.isnan(W).any() or torch.isnan(mu).any() or torch.isnan(sigma).any():
        print("Debug ELBO_KL_divergence_Monte_Carlo:")
        print(f"W range: [{W.min():.4f}, {W.max():.4f}]")
        print(f"mu range: [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    
    # log q and log p
    log_q = -0.5 * (((W - mu) / sigma_safe) ** 2 + torch.log(2 * torch.pi * sigma_safe**2))
    log_p = -0.5 * (W**2 + torch.log(2 * torch.pi))
    
    # Debug prints if NaN
    if torch.isnan(log_q).any() or torch.isnan(log_p).any():
        print(f"log_q range: [{log_q.min():.4f}, {log_q.max():.4f}]")
        print(f"log_p range: [{log_p.min():.4f}, {log_p.max():.4f}]")
    
    # sum over dims, mean over samples
    return torch.mean(torch.sum(log_q - log_p, dim=1))

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

# TODO: Implement as however you require this!
# TODO: Make sure to use the finally found optimal parameters once you submit your solution.

# Initialize with small values to prevent instability
D_plus1 = X_train_t.shape[1] + 1
# Initialize as leaf tensors with small random values
mu_param = torch.nn.Parameter(torch.randn(D_plus1) * 0.01)  # Very small random values
log_sigma_param = torch.nn.Parameter(torch.full((D_plus1,), -2.0))  # Start with sigma ≈ 0.135

# Use a much smaller learning rate and add weight decay
optimizer = torch.optim.Adam([
    {'params': mu_param, 'lr': 1e-5},
    {'params': log_sigma_param, 'lr': 1e-5}
], weight_decay=1e-4)

EPOCHS = 50
USE_KL_TYPE: KL_DIV_TYPE = 'analytical'
elbo_history = []

for epoch in range(EPOCHS):
    epoch_elbo = 0.0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        
        # Ensure sigma is positive and not too small/large
        sigma = torch.exp(log_sigma_param).clamp(min=1e-2, max=1.0)
        eps = torch.randn(NUM_VARIATIONAL_SETS, D_plus1) * 0.1  # Scale down noise
        
        # Debug prints for first batch of first epoch
        if epoch == 0 and len(elbo_history) == 0:
            print("Debug values:")
            print(f"mu_param range: [{mu_param.min():.4f}, {mu_param.max():.4f}]")
            print(f"sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
            print(f"eps range: [{eps.min():.4f}, {eps.max():.4f}]")
            print(f"Xb range: [{Xb.min():.4f}, {Xb.max():.4f}]")
            print(f"yb range: [{yb.min():.4f}, {yb.max():.4f}]")
        
        # Calculate ELBO components separately for debugging
        W = mu_param.unsqueeze(0) + sigma.unsqueeze(0) * eps
        exp_lik = ELBO_expected_data_likelihood(yb, Xb, W)
        kl_div = ELBO_KL_divergence_analytical(mu_param, sigma)
        elbo = exp_lik - kl_div
        loss = -elbo
        
        # Debug prints for first batch of first epoch
        if epoch == 0 and len(elbo_history) == 0:
            print(f"exp_lik: {exp_lik.item():.4f}")
            print(f"kl_div: {kl_div.item():.4f}")
            print(f"elbo: {elbo.item():.4f}")
            print(f"loss: {loss.item():.4f}")
        
        # Check for NaN before backward pass
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch}")
            print(f"mu_param: {mu_param}")
            print(f"sigma: {sigma}")
            print(f"exp_lik: {exp_lik}")
            print(f"kl_div: {kl_div}")
            print(f"elbo: {elbo}")
            # Reset parameters to safe values
            with torch.no_grad():
                mu_param.data.zero_()
                log_sigma_param.data.fill_(-2.0)
            continue
            
        loss.backward()
        
        # Debug prints for first batch of first epoch
        if epoch == 0 and len(elbo_history) == 0:
            print(f"mu_param grad range: [{mu_param.grad.min():.4f}, {mu_param.grad.max():.4f}]")
            print(f"log_sigma_param grad range: [{log_sigma_param.grad.min():.4f}, {log_sigma_param.grad.max():.4f}]")
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_([mu_param, log_sigma_param], max_norm=0.1)
        
        optimizer.step()
        
        # Ensure parameters stay in reasonable ranges after update
        with torch.no_grad():
            mu_param.data.clamp_(min=-1.0, max=1.0)
            log_sigma_param.data.clamp_(min=-5.0, max=0.0)
        
        epoch_elbo += elbo.item()
    
    # Check for NaN in epoch ELBO
    if torch.isnan(torch.tensor(epoch_elbo)):
        print(f"NaN detected in epoch ELBO at epoch {epoch}")
        continue
        
    elbo_history.append(epoch_elbo / len(train_loader))
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} ELBO: {elbo_history[-1]:.4f}")
        # Print current parameter ranges
        print(f"mu_param range: [{mu_param.min():.4f}, {mu_param.max():.4f}]")
        print(f"sigma range: [{torch.exp(log_sigma_param).min():.4f}, {torch.exp(log_sigma_param).max():.4f}]")

# Final variational parameters
final_mu = mu_param.detach()
final_sigma = torch.exp(log_sigma_param).detach()
print("Final mu:", final_mu)
print("Final sigma:", final_sigma)

# Plot training ELBO
plt.figure(figsize=(10, 6))
plt.plot(elbo_history)
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.title('ELBO over Training')
plt.grid(True)
plt.show()

# ELBO on test data
eps_test = torch.randn(NUM_VARIATIONAL_SETS, D_plus1)
test_elbo = ELBO(mu_param, torch.exp(log_sigma_param).clamp(min=1e-6), y_test_t, X_test_t, eps_test, kl=USE_KL_TYPE)
print(f"Test set ELBO: {test_elbo:.4f}")