import numpy as np
import math

np.random.seed(16)

# --- Faster RBF using norms  ---
def rbf_kernel(X1, X2, length_scale):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    X1_sq = np.sum(X1**2, axis=1)[:, None]
    X2_sq = np.sum(X2**2, axis=1)[None, :]
    sqdist = np.maximum(X1_sq + X2_sq - 2.0 * (X1 @ X2.T), 0.0)
    return np.exp(-0.5 * sqdist / (length_scale**2))

# Standard Normal PDF/CDF
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

# --- GP posterior returning mean and *diagonal* variance only (latent f) ---
def gp_posterior_mean_var(X_train, y_train, X_test, alpha, length_scale):
    K = rbf_kernel(X_train, X_train, length_scale) + (alpha**2 + 1e-8) * np.eye(len(X_train))
    L = np.linalg.cholesky(K)
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    K_s = rbf_kernel(X_train, X_test, length_scale)
    mu = K_s.T @ alpha_vec
    v = np.linalg.solve(L, K_s)
    var = 1.0 - np.sum(v**2, axis=0)          
    var = np.maximum(var, 1e-12)               # numerical safety
    return mu, var

# --- EI for MINIMIZATION  ---
def expected_improvement_vectorized(mu, sigma, f_best, xi):
    sigma = np.maximum(sigma, 1e-12)
    Z = (f_best - mu - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    return (f_best - mu - xi) * Phi + sigma * phi

def initialize_trust_region(bounds, initial_radius):
    trust_region_center = None
    trust_region_radius = initial_radius
    return trust_region_center, trust_region_radius

# --- Trust-region update: recenter only on success; otherwise stay at previous best ---
def update_trust_region(trust_region_center, trust_region_radius,
                        X_new, y_new, y_best_prev, X_best_prev,
                        bounds, shrink_factor, expand_factor, tol=0.0):

    success = (y_new < y_best_prev - tol)
    if success:
        trust_region_radius *= expand_factor
        trust_region_center  = X_new
    else:
        trust_region_radius *= shrink_factor
        trust_region_center  = X_best_prev

    span = np.ptp(bounds, axis=1)
    min_radius = 0.01 * span
    max_radius = 0.5  * span
    trust_region_radius = np.clip(trust_region_radius, min_radius, max_radius)
    return trust_region_center, trust_region_radius

# --- Candidate selection inside the trust region ---
def random_acquisition_maximization(X_train, y_train, trust_region_center, trust_region_radius, 
                                    bounds, num_samples, alpha, length_scale, xi):
    dim = bounds.shape[0]
    lower_bounds = np.maximum(trust_region_center - trust_region_radius, bounds[:, 0])
    upper_bounds = np.minimum(trust_region_center + trust_region_radius, bounds[:, 1])
    samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_samples, dim))

    mu, var = gp_posterior_mean_var(X_train, y_train, samples, alpha=alpha, length_scale=length_scale)
    sigma = np.sqrt(var)
    f_best = np.min(y_train)  # minimization
    ei = expected_improvement_vectorized(mu, sigma, f_best, xi)
    idx_max = np.argmax(ei)
    return samples[idx_max]

def bayesian_optimization_with_trust_region_fast(n_iters, sample_loss, bounds, n_pre_samples,
                                                 alpha=0.1, initial_trust_radius=0.1, length_scale=2,
                                                 xi=0.01, shrink_factor=0.8, expand_factor=1.3,
                                                 num_samples=1000, verbose=True, tol=0.0):
    bounds = np.array(bounds)
    dim = bounds.shape[0]

    # Initial design
    X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_pre_samples, dim))
    y_train = np.array([sample_loss(x) for x in X_train])

    # Initialize TR at current best
    best_idx = np.argmin(y_train)
    trust_region_center = X_train[best_idx].copy()
    trust_region_radius = np.full(dim, initial_trust_radius)

    iteration_numbers = []

    for i in range(n_iters):

        X_next = random_acquisition_maximization(
            X_train, y_train, trust_region_center, trust_region_radius, bounds,
            num_samples, alpha, length_scale, xi
        )


        best_prev_idx = np.argmin(y_train)
        y_best_prev   = y_train[best_prev_idx]
        X_best_prev   = X_train[best_prev_idx].copy()

        # Evaluate
        y_next = sample_loss(X_next)

        # Update dataset
        X_train = np.vstack((X_train, X_next.reshape(1, -1)))
        y_train = np.append(y_train, y_next)


        trust_region_center, trust_region_radius = update_trust_region(
            trust_region_center, trust_region_radius,
            X_next, y_next, y_best_prev, X_best_prev,
            bounds, shrink_factor, expand_factor, tol=tol
        )

        iteration_numbers.append(i + 1)

        if verbose:
            print(f"Iteration {i+1}/{n_iters}, X_next = {X_next}, y_next = {y_next}, "
                  f"TR center = {trust_region_center}, radius = {trust_region_radius}")

    return X_train, y_train, iteration_numbers

