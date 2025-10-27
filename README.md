# **Gaussian Process Bayesian Optimization (GPBO) Model**

Note: At the time of this project i was at the start of my third year chemical engineering degree and in a lab decided to apply Bayesian Optimisation. Initialy due to lack of time I was not able to develop my own bayesian optimisation tool so this work is an expansion on my previous work which allows for a good model comparison. 



---

Welcome to the documentation of the Gaussian Process Bayesian Optimization (GPBO) model. This repository provides an implementation of Bayesian Optimization using Gaussian Processes, with a focus on optimizing functions where evaluations are expensive. The model includes a Trust Region approach to balance exploration and exploitation efficiently.

In 

---

## **Introduction**

Bayesian Optimization is a powerful technique for optimizing objective functions that are expensive to evaluate. It is particularly useful when:

- The function lacks an analytical expression.
- The function evaluation is costly (e.g., requires running complex simulations or experiments).
- Derivatives of the function are unavailable.

Gaussian Processes (GP) provide a probabilistic approach to modeling the objective function, capturing both the mean and uncertainty of predictions. By leveraging GPs within Bayesian Optimization, we can make informed decisions about where to sample next, balancing the trade-off between exploration (sampling where uncertainty is high) and exploitation (sampling where the mean prediction is optimal).

This implementation enhances the standard Bayesian Optimization by incorporating a **Trust Region** method, which dynamically adjusts the search space based on the progress of the optimization.

---

## **Code Deatials**
The repository has the following structure

```
BAYESIAN-OPTIMISATION-WITH-...                        
├── Main/
│   ├── __pycache__/                    # Python bytecode (auto-generated)
│   ├── GPBO.py                         # Gaussian-Process Bayesian Optimisation (core implementation)
│   ├── GPBO2.py                        # Alternative/experimental GP-BO implementation or API variant
│   ├── vanGPBO.py                      # “Vanilla” GP-BO baseline (minimal features for comparison)
│   └── Test.ipynb                      # Notebook comparing variants/benchmarks of GP-BO
├── LICENSE                             # Project license
├── README.md                           # This documentation
└── Requirements.txt                    # Python dependencies
```

### What each file/module is for

* **Main/vanGPBO.py** – Baseline “vanilla” GP-BO to provide a simple reference against the implementations.
* **Main/GPBO.py** – Primary Gaussian-Process Bayesian Optimisation module initially formulated
* **Main/GPBO2.py** – Second implementation to speed up the training
* **Main/Test.ipynb** – Side-by-side comparisons/plots of the different GP-BO variants on 
* **Requirements.txt** – install the Python packages needed to run the project.
* **LICENSE** – Legal terms for using the code


## **Prerequisites**

- Python 3.6 or higher
- NumPy
- Matplotlib (for visualization)
- SciPy (optional, for advanced optimization techniques)
- scikit-learn (for comparison purposes)

---

## **Installation**

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/barbacci_marco/Trust-Region-BO.git

```

Install the required packages:

```bash
pip install numpy 
```

### Quick start

Use the notebook `Main/GPBO_compare.ipynb` to visualise and compare the behaviour of `GPBO.py`, `GPBO2.py`, and `vanGPBO.py` after generating results.

## **Implementation Details**

The GPBO implementation is modular and centered on a Gaussian Process surrogate with a trust-region strategy and random acquisition maximization.

### **Kernel: RBF (Squared Exponential)**

```python
def rbf_kernel(X1, X2, length_scale):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
    return np.exp(-0.5 / length_scale**2 * sqdist)
```

* **Parameters**

  * `length_scale`: Controls smoothness. Smaller values allow faster variation.


### **Standard Normal Utilities**

The code includes a fast error-function approximation and convenience wrappers:

```python
def standard_normal_pdf(z): ...
def erf(z): ...
def standard_normal_cdf(z): ...
```

These are used in the Expected Improvement calculation.

### **Gaussian Process Posterior**

Noise is applied **only to the training covariance `K`** . Jitter is added for stability and the posterior covariance is symmetrized and clipped on the diagonal.

```python
def gp_posterior(X_train, y_train, X_test, alpha, length_scale):
    K   = rbf_kernel(X_train, X_train, length_scale) + (alpha**2) * np.eye(len(X_train))
    K  += 1e-8 * np.eye(len(K))  # jitter

    K_s  = rbf_kernel(X_train, X_test,  length_scale)
    K_ss = rbf_kernel(X_test,  X_test,  length_scale)

    L = np.linalg.cholesky(K)
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    mu_s  = K_s.T.dot(alpha_vec)
    v     = np.linalg.solve(L, K_s)
    cov_s = K_ss - v.T.dot(v)

    cov_s = 0.5 * (cov_s + cov_s.T)
    diag = np.clip(np.diag(cov_s), 0.0, None)
    np.fill_diagonal(cov_s, diag)

    return mu_s.flatten(), cov_s
```



### **Acquisition: Expected Improvement (EI, minimization form)**

Formulated for **minimization**: improvement vs. the best *low* value.

```python
def expected_improvement(X, X_train, y_train, mu, sigma, f_best, xi):
    sigma = np.maximum(sigma, 1e-12)
    Z   = (f_best - mu - xi) / sigma
    phi = standard_normal_pdf(Z)
    Phi = standard_normal_cdf(Z)
    return (f_best - mu - xi) * Phi + sigma * phi
```

* **Parameters**

  * `xi`: Exploration parameter (higher ⇒ more exploration).

### **Trust Region Strategy**

* **Initialize**

  ```python
  def initialize_trust_region(bounds, initial_radius):
      trust_region_center = None
      trust_region_radius = initial_radius
      return trust_region_center, trust_region_radius
  ```
* **Update**
  On success (`y_new < y_best_prev - tol`), **expand** and center at `X_new`. Otherwise, **shrink** and recenter at previous best `X_best_prev`. Per-dimension radii are clamped to `[1% span, 50% span]` of the bounds.

  ```python
  def update_trust_region(trust_region_center, trust_region_radius,
                          X_new, y_new, y_best_prev, X_best_prev, bounds,
                          shrink_factor, expand_factor, tol=0.0):
      ...
      return trust_region_center, trust_region_radius
  ```

### **Random Acquisition Maximization**

Samples uniformly within the current trust region, evaluates EI using the GP posterior at those samples, and picks the argmax.

```python
def random_acquisition_maximization(acquisition, X_train, y_train,
                                    trust_region_center, trust_region_radius,
                                    bounds, num_samples, alpha, length_scale, xi):
    ...
    return X_next
```

### **Main Loop**

```python
def bayesian_optimization_with_trust_region(
    n_iters, sample_loss, bounds, n_pre_samples,
    alpha=0.1, initial_trust_radius=0.1, length_scale=2, xi=0.01,
    shrink_factor=0.8, expand_factor=1.3, num_samples=1000):
    ...
    return X_train, y_train, iteration_numbers
```

## **References**
- E. A. del Rio Chanona, P. Petsagkourakis, E. Bradford, J. E. Alves Graciano, B. Chachuat,
Real-time optimization meets Bayesian optimization and derivative-free optimization: A tale of modifier adaptation, Computers & Chemical Engineering, Volume 147, 2021, 107249, ISSN 0098-1354, https://doi.org/10.1016/j.compchemeng.2021.107249.
(https://www.sciencedirect.com/science/article/pii/S0098135421000272)
- Dürholt, J.P. et al. (2024) BoFire: Bayesian Optimization Framework intended for real experiments, arXiv.org. Available at: https://arxiv.org/abs/2408.05040 (Accessed: 01 November 2024). 
---

## **License**

This project is licensed under the MIT License - see the license file for details.

---

**Note:** For any questions or contributions, please open an issue or submit a pull request.
