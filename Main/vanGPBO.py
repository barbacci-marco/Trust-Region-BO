import numpy as np
import math

np.random.seed(16)

# --- RBF Kernel ---
def rbf_kernel(X1, X2, length_scale):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
    return np.exp(-0.5 / length_scale**2 * sqdist)

# --- Standard normal PDF/CDF  ---
def standard_normal_pdf(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

def erf(z):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911
    sign = np.sign(z)
    z = np.abs(z)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    return sign * y

def standard_normal_cdf(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))

# --- GP posterior (latent f): noise on K only---
def gp_posterior(X_train, y_train, X_test, alpha, length_scale):
    K   = rbf_kernel(X_train, X_train, length_scale) + (alpha**2) * np.eye(len(X_train))
    K  += 1e-8 * np.eye(len(K))  # jitter
    K_s = rbf_kernel(X_train, X_test,  length_scale)
    K_ss = rbf_kernel(X_test,  X_test,  length_scale)

    L = np.linalg.cholesky(K)
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    mu_s = K_s.T.dot(alpha_vec)
    v = np.linalg.solve(L, K_s)
    cov_s = K_ss - v.T.dot(v)

    
    cov_s = 0.5 * (cov_s + cov_s.T)
    diag = np.clip(np.diag(cov_s), 0.0, None)
    np.fill_diagonal(cov_s, diag)

    return mu_s.flatten(), cov_s

# --- Expected Improvement for MIN ---
def expected_improvement(mu, sigma, f_best, xi):
    sigma = np.maximum(sigma, 1e-12)
    Z = (f_best - mu - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    return (f_best - mu - xi) * Phi + sigma * phi


def random_acquisition_maximization_global(X_train, y_train, bounds, num_samples, alpha, length_scale, xi):
    bounds = np.asarray(bounds)
    dim = bounds.shape[0]
    samples = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_samples, dim))

    mu, cov = gp_posterior(X_train, y_train, samples, alpha=alpha, length_scale=length_scale)
    var = np.clip(np.diag(cov), 0.0, None)
    sigma = np.sqrt(var)
    f_best = np.min(y_train)

    ei = expected_improvement(mu, sigma, f_best, xi)
    return samples[np.argmax(ei)]


def bayesian_optimization(n_iters, sample_loss, bounds, n_pre_samples,
                          alpha=0.1, length_scale=2.0, xi=0.01, num_samples=1000):
    bounds = np.array(bounds)
    dim = bounds.shape[0]

    # Initial design
    X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_pre_samples, dim))
    y_train = np.array([sample_loss(x) for x in X_train])

    iteration_numbers = []

    for i in range(n_iters):
        # Choose next point globally by maximizing EI over random candidates
        X_next = random_acquisition_maximization_global(
            X_train, y_train, bounds, num_samples, alpha, length_scale, xi
        )

        y_next = sample_loss(X_next)

        # Update dataset
        X_train = np.vstack((X_train, X_next.reshape(1, -1)))
        y_train = np.append(y_train, y_next)

        iteration_numbers.append(i + 1)

        print(f"Iteration {i+1}/{n_iters}, X_next = {X_next}, y_next = {y_next}")

    return X_train, y_train, iteration_numbers
