import numpy as np
import math

np.random.seed(16)

# RBF Kernel Function
def rbf_kernel(X1, X2, length_scale):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
    return np.exp(-0.5 / length_scale**2 * sqdist)

# Standard Normal PDF
def standard_normal_pdf(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

# Error Function Approximation
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

# Standard Normal CDF
def standard_normal_cdf(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))

# GP Posterior Function 
def gp_posterior(X_train, y_train, X_test, alpha, length_scale):

    K   = rbf_kernel(X_train, X_train, length_scale) + (alpha**2) * np.eye(len(X_train))
    K  += 1e-8 * np.eye(len(K))  # jitter for numerical stability


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

def expected_improvement(X, X_train, y_train, mu, sigma, f_best, xi):
    sigma = np.maximum(sigma, 1e-12)
    Z = (f_best - mu - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    ei = (f_best - mu - xi) * Phi + sigma * phi
    return ei

def initialize_trust_region(bounds, initial_radius):
    trust_region_center = None
    trust_region_radius = initial_radius
    return trust_region_center, trust_region_radius

def update_trust_region(trust_region_center, trust_region_radius,
                        X_new, y_new, y_best_prev, X_best_prev, bounds,
                        shrink_factor, expand_factor, tol=0.0):

    success = (y_new < y_best_prev - tol)

    if success:
        trust_region_radius *= expand_factor
        trust_region_center  = X_new
    else:
        trust_region_radius *= shrink_factor
        trust_region_center  = X_best_prev

    # Clamp per-dimension
    min_radius = 0.01 * np.ptp(bounds, axis=1)
    max_radius = 0.5  * np.ptp(bounds, axis=1)
    trust_region_radius = np.maximum(np.minimum(trust_region_radius, max_radius), min_radius)

    return trust_region_center, trust_region_radius

def random_acquisition_maximization(acquisition, X_train, y_train, trust_region_center, trust_region_radius, 
                                    bounds, num_samples, alpha, length_scale, xi):
    lower_bounds = np.maximum(trust_region_center - trust_region_radius, bounds[:, 0])
    upper_bounds = np.minimum(trust_region_center + trust_region_radius, bounds[:, 1])
    samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_samples, bounds.shape[0]))
    
    # Compute GP posterior for samples
    mu, cov = gp_posterior(X_train, y_train, samples, alpha=alpha, length_scale=length_scale)
    # Numerical safety on variance
    var = np.clip(np.diag(cov), 0.0, None)
    sigma = np.sqrt(var)
    f_best = np.min(y_train)  # Minimization problem
    
    # Compute acquisition values
    acquisition_values = np.array([
        acquisition(x.reshape(1, -1), X_train, y_train, mu_i, sigma_i, f_best, xi)
        for x, mu_i, sigma_i in zip(samples, mu, sigma)
    ])
    
    # Select the point with the highest acquisition value
    idx_max = np.argmax(acquisition_values)
    X_next = samples[idx_max]
    
    return X_next

def bayesian_optimization_with_trust_region(n_iters, sample_loss, bounds, n_pre_samples, alpha=0.1, initial_trust_radius=0.1, length_scale=2, xi=0.01, shrink_factor=0.8, expand_factor=1.3, num_samples=1000):
   
    bounds = np.array(bounds)
    X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_pre_samples, bounds.shape[0]))
    y_train = np.array([sample_loss(x) for x in X_train])
    
    y_best_idx = np.argmin(y_train)
    trust_region_center = X_train[y_best_idx]
    trust_region_radius = initial_trust_radius * np.ones(bounds.shape[0])
    
    iteration_numbers = []
    
    for i in range(n_iters):
        # (Optional posterior on training points not needed for math; kept as-is)
        mu_s, cov_s = gp_posterior(X_train, y_train, X_train, alpha, length_scale)
        
        best_prev_idx = np.argmin(y_train)
        y_best_prev   = y_train[best_prev_idx]
        X_best_prev   = X_train[best_prev_idx]
        
        X_next = random_acquisition_maximization(
            expected_improvement, X_train, y_train,
            trust_region_center, trust_region_radius, bounds,
            num_samples, alpha, length_scale, xi
        )

        # Save previous best BEFORE adding the new point 
        y_best_prev = np.min(y_train)
        
        y_next = sample_loss(X_next)

        X_train = np.vstack((X_train, X_next.reshape(1, -1)))
        y_train = np.append(y_train, y_next)

        trust_region_center, trust_region_radius = update_trust_region(
            trust_region_center, trust_region_radius,
            X_next, y_next, y_best_prev, X_best_prev, bounds,
            shrink_factor, expand_factor, tol=1e-6  
        )
        
        iteration_numbers.append(i + 1)
        
        print(f"Iteration {i+1}/{n_iters}, X_next = {X_next}, y_next = {y_next}, "
              f"Trust Region Center = {trust_region_center}, Radius = {trust_region_radius}")
    
    return X_train, y_train, iteration_numbers

