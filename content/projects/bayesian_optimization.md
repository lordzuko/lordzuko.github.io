---
title: "Bayesian Optimization using Gaussian Processes"
description: "Bayesian learning in action"
draft: false
dateString: June 2023
tags: ["Python", "Bayesian Learning", "Gaussian Process"]
showToc: true
weight: 205
# cover:
#     image: "projects/automated-image-captioning/cover.jpg"
--- 

### 🔗 [Github](https://github.com/lordzuko/SpeakingStyle)

## Description

## Implementation

```python:
def rmse(y_true, y_label):
    N = len(y_true)
    return np.sqrt(np.mean((y_true - y_label)**2))

def log_rmse(y_true, y_label):
    return np.log(rmse(y_true, y_label))

## Methods for training NN
def random_initialization_weights(D, K):
    np.random.seed(1)
    ww = np.random.randn(K) / np.sqrt(K)
    bb = np.random.randn(1)/np.sqrt(K)
    V = np.random.randn(K,D) / np.sqrt(K)
    bk = np.random.randn(K) / np.sqrt(K)
    return ww, bb, V, bk

def fit_nn_gradopt(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
#     init = (np.zeros(K), np.array(0), np.zeros((K,D)), np.zeros(K))
    init = random_initialization_weights(D, K)
    ww_bar, bb_bar, V_bar, bk_bar  = minimize_list(nn_cost, init, args)
    return ww_bar, bb_bar, V_bar, bk_bar


def train_nn_reg(alpha):
    #fit model
    ww, bb, V, bk = fit_nn_gradopt(X_train, y_train, alpha)
    # calculate log rmse on validation data
    error_nn_valid = log_rmse(y_val, nn_cost((ww, bb, V, bk), X_val))
    # val_4a_baseline = np.log(0.27248788663267565)
    # return -(error_nn_valid - val_4a_baseline)
    return error_nn_valid
```

```python:

def improvement_probability(x_test, 
                            X_obs, 
                            y_obs, 
                            sigma_y=0.05, 
                            ell=5.0, 
                            sigma_f=0.1):
    """
    This function calculated probability of improvement
    """
    x_test = x_test 
    X_obs = X_obs 

    post_mean, post_cov = gp_post_par(x_test, X_obs, y_obs, sigma_y, ell, sigma_f)
    std = np.sqrt(np.diag(post_cov))
    
    delta = (post_mean - np.max(y_obs))
    # added 1e-9 to avoid division by zero
    z = delta / (std + 1e-9)
    return norm.cdf(z)

# posterior
def sample_from_gaussian(size, 
                         mean, 
                         cov):

    L = np.linalg.cholesky(post_cov)
    return np.dot(L, np.random.normal(size=size)) + mean[:, None]

# posterior predictive
def get_predictions_2_std(x, 
                          X_obs, 
                          y_obs, 
                          sigma_y=0.05, 
                          ell=5.0, 
                          sigma_f=0.1):
    if x.shape == ():
        x = np.array([x])

    post_mean, post_cov = gp_post_par(x, X_obs, y_obs, sigma_y, ell, sigma_f)
    post_std = np.sqrt(np.diag(post_cov))
    return post_mean, post_mean - 2*post_std, post_mean + 2*post_std

def sample_next_simple(n_samples, 
                       acquisition_fn, 
                       param_choices, 
                       X_obs, 
                       y_obs, 
                       bounds,
                       sigma_y=0.05, 
                       ell=5.0, 
                       sigma_f=0.1):
    
    best_ei = np.inf
    best_x_test = None
    
    sampled_x = sorted(param_choices)
    sampled_y = []
    sampled_y_1 = []
    sampled_y_2 = []
    sampled_pi = []
    
    for i, param in enumerate(sampled_x):
        opt = minimize(fun=acquisition_fn,
                 x0=np.array([param]),
                 bounds=bounds,
                 args=(X_obs, y_obs, sigma_y,ell,sigma_f))
        
        #sample from gaussian
        y, y1, y2 = get_predictions_2_std(param, 
                                          X_obs, 
                                          y_obs,
                                          sigma_y, 
                                          ell, 
                                          sigma_f)
#         sampled_y.append(sample_from_gaussian(param, X_obs, y_obs)[0])
        sampled_y.append(y[0])
        sampled_y_1.append(y1[0])
        sampled_y_2.append(y2[0])
        sampled_pi.append(-opt.fun)
        
        if opt.fun < best_ei:
            best_ei = opt.fun
            best_x_test = opt.x
        #   print(i, best_x_test, best_ei)
    
    #    print(f"Best in this iter: {best_x_test} {best_ei}")
    #    print("______________")
    sampled_data = {"x": sampled_x, 
                    "y": np.array(sampled_y), 
                    "y1":np.array(sampled_y_1),
                    "y2":np.array(sampled_y_2),
                    "pi": np.array(sampled_pi)}
    
    return best_x_test, - best_ei, sampled_data
```

```python:
def acquisition_fn(x_test, X_obs, y_obs,sigma_y, ell,sigma_f):
    return -improvement_probability(x_test, X_obs, y_obs, sigma_y, ell,sigma_f)
    

# https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization/
def bayesian_optimization(n_iters, 
                          loss_function, 
                          alpha_min=0, 
                          alpha_max=50, 
                          n_samples=50,
                          sigma_y=0.05, 
                          ell=5.0,
                          sigma_f=0.1,
                          random_seed=1):
    np.random.seed(random_seed)
    track_metrics = {}
    # Consider alpha in range (0,50, 0.02)
    # Pick three values from this as training locations
    # Use train_nn_reg to get y_train on these values

    X_grid = np.arange(alpha_min, alpha_max,0.02)
    N_grid = X_grid.shape[0]

    # Select the first three training points
    #     idx = np.round(N_grid * np.array([0.33, 0.66, 0.99])).astype(int)
    idx = np.random.choice(list(range(N_grid)), 3, replace=False)
    X_obs = X_grid[sorted(idx)]

    y_obs = []
    # These are the alpha params from which we will sample for 
    # optimizing the acquisition function
    param_choices = np.delete(X_grid, idx, 0)

    val_4a_baseline = np.log(0.27439047802923244)
    
    for alpha in X_obs:
        y_obs.append(train_nn_reg(alpha))

    # this is as suggested in the project instructions
    y_obs = np.array(y_obs) - val_4a_baseline
    
    track_metrics[0] = {
        "obs": {
            "x": X_obs,
            "y": y_obs
        },
        "sampled_data": {
            "x": X_grid
        }
        
    }
    track_metrics[0]["sampled_data"]["y"] = []
    track_metrics[0]["sampled_data"]["y1"] = []
    track_metrics[0]["sampled_data"]["y2"] = []
    
    for t in track_metrics[0]["sampled_data"]["x"]:
        z, z_1, z_2 = get_predictions_2_std(t, X_obs, y_obs, sigma_y, ell,sigma_f)
        track_metrics[0]["sampled_data"]["y"].append(z[0])
        track_metrics[0]["sampled_data"]["y1"].append(z_1[0])
        track_metrics[0]["sampled_data"]["y2"].append(z_2[0])
        
    track_metrics[0]["sampled_data"]["y"] = np.array(track_metrics[0]["sampled_data"]["y"])
    track_metrics[0]["sampled_data"]["y1"] = np.array(track_metrics[0]["sampled_data"]["y1"])
    track_metrics[0]["sampled_data"]["y2"] = np.array(track_metrics[0]["sampled_data"]["y2"])
    
    
    for i in range(n_iters):
        #   Less Expensive optimization step
        track_metrics[i+1] ={}
        x_next_best, ei_value, sampled_data = sample_next_simple(n_samples, 
                                                                 acquisition_fn, 
                                                                 param_choices,
                                                                 X_obs, 
                                                                 y_obs, 
                                                                 bounds=[(alpha_min, alpha_max)],
                                                                 sigma_y=sigma_y, 
                                                                 ell=ell,
                                                                 sigma_f=sigma_f)
        
        # Expensive true objective evaluation steps
        y_next_best = loss_function(alpha=x_next_best[0])
        
        # Do not reselect already observed points
        # param_choices = np.delete(param_choices, param_choices == x_next_best, axis=0)
        
        X_obs = np.append(X_obs, x_next_best)
        y_obs = np.append(y_obs, [y_next_best])

        
        track_metrics[i+1] = {
            "obs": {"x": X_obs,
                    "y": y_obs},
            "alpha": x_next_best[0],
            "pi": ei_value,
            "fn_val": y_next_best,
            "sampled_data": sampled_data
        }
    
    
    return track_metrics
```

```python:
metrics_setting1 = bayesian_optimization(6, train_nn_reg)
for iteration in range(1, 6):
    print(f"Iteration: {iteration} Alpha: {metrics_setting1[iteration]['alpha']} PI: {metrics_setting1[iteration]['pi']} ")
```

```python:
alpha = metrics_setting1[5]['alpha']
ww, bb, V, bk = fit_nn_gradopt(X_train, y_train, alpha)

# calculate log rmse on validation data
rmse_valid = rmse(y_val, nn_cost((ww, bb, V, bk), X_val))
rmse_test = rmse(y_test, nn_cost((ww, bb, V, bk), X_test))

print(f"RMSE Valid: {rmse_valid}")
print(f"RMSE Test: {rmse_valid}")
```

```python:
# Idea for plots is inspired form this blog:
# https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization/

for k, v in metrics_setting1.items():
        
    #     idx = np.argsort(v["sampled_data"]["x"])
    x = v["sampled_data"]["x"]
    y = np.array(v["sampled_data"]["y"])
    y1 = np.array(v["sampled_data"]["y1"])
    y2 = np.array(v["sampled_data"]["y2"])
    if k != 0:
        pi = np.array(v["sampled_data"]["pi"])
#     plt.plot(x, y, "-")
#     plt.plot(v["obs"]["x"], v["obs"]["y"], "x")
    fig, axs = plt.subplots(1,2, figsize=(20,4))
    plt.suptitle(f"Iteration {k}")
    axs[0].set(xlabel="alpha", ylabel="f", xlim=(-1,51), ylim=(-2, 0.5))

    # plot observation points
    axs[0].scatter(v["obs"]["x"][:3], v["obs"]["y"][:3], color="r", label="random init")
    axs[0].scatter(v["obs"]["x"][3:], v["obs"]["y"][3:], color="g", label="proposed params")

    # plot funs
    if k == 0:
        axs[0].set_title(f' Initial Fn: {k}')
    else:
        axs[0].set_title(f'{k} - Best-Alpha: {round(v["alpha"],2)} Best Fn: {round(v["fn_val"], 4)}')
    axs[0].plot(x, y, '-k', linewidth=2, label='mean')
    axs[0].plot(x, y1, '--y', linewidth=1, label="-2 std")
    axs[0].plot(x, y2, '--c', linewidth=1, label="+2 std")
    axs[0].fill_between(x, y1, y2, label="uncertainty", alpha=0.1, color="#dddddd")
        
    axs[0].legend()
        
    # plot probability improvement v/s alpha
    if k != 0:
        axs[1].set(xlabel="alpha", ylabel="prob improvement (PI)", xlim=(-1,51))
        axs[1].set_title(f'{k} - Best-Alpha: {round(v["alpha"], 2)} Best PI: {round(v["pi"],4)}')
        axs[1].vlines(v["alpha"], 0, v["pi"], linestyle="--", color="r", label="New proposal")
        axs[1].plot(x, pi, '-')
    
    axs[1].legend()
    plt.subplots_adjust(hspace=0.5)
    
```
### Probability of Improvement
```
def improvement_probability(x_test, 
                            X_obs, 
                            y_obs, 
                            sigma_y=0.05, 
                            ell=5.0, 
                            sigma_f=0.1):
    """
    This function calculated probability of improvement
    """
    x_test = x_test 
    X_obs = X_obs 

    post_mean, post_cov = gp_post_par(x_test, X_obs, y_obs, sigma_y, ell, sigma_f)
    std = np.sqrt(np.diag(post_cov))
    
    delta = (post_mean - np.max(y_obs))
    # added 1e-9 to avoid division by zero
    z = delta / (std + 1e-9)
    return norm.cdf(z)
```
|                                                              |
| :----------------------------------------------------------: |
| ![my image](images/projects/bayesian-optimization/poi/0.png) |
| ![my image](images/projects/bayesian-optimization/poi/1.png) |
| ![my image](images/projects/bayesian-optimization/poi/2.png) |
| ![my image](images/projects/bayesian-optimization/poi/3.png) |
| ![my image](images/projects/bayesian-optimization/poi/4.png) |
| ![my image](images/projects/bayesian-optimization/poi/5.png) |
| ![my image](images/projects/bayesian-optimization/poi/6.png) |


### Expected Improvement
```python:
def expected_improvement(x_test, 
                            X_obs, 
                            y_obs, 
                            ex_ei=0.01,
                            sigma_y=0.05, 
                            ell=5.0, 
                            sigma_f=0.1):
    """
    This function calculates expected improvement
    """
    x_test = x_test 
    X_obs = X_obs 

    post_mean, post_cov = gp_post_par(x_test, X_obs, y_obs, sigma_y, ell, sigma_f)
    std = np.sqrt(np.diag(post_cov))
    
    delta = (post_mean - np.max(y_obs) - ex_ei)
    # added 1e-9 to avoid division by zero
    z = delta / (std + 1e-9)
    exp_improv = delta * norm.cdf(z) + std * norm.pdf(z)
    exp_improv[std == 0.0] = 0.0
    
    return exp_improv
```
|                                                             |
| :---------------------------------------------------------: |
| ![my image](images/projects/bayesian-optimization/ei/0.png) |
| ![my image](images/projects/bayesian-optimization/ei/1.png) |
| ![my image](images/projects/bayesian-optimization/ei/2.png) |
| ![my image](images/projects/bayesian-optimization/ei/3.png) |
| ![my image](images/projects/bayesian-optimization/ei/4.png) |
| ![my image](images/projects/bayesian-optimization/ei/5.png) |
| ![my image](images/projects/bayesian-optimization/ei/6.png) |

### Lower Confidence Bound

```python:
def lower_confidence_bound(x_test, 
                           X_obs, 
                           y_obs, 
                           k=0.01,
                           sigma_y=0.05, 
                           ell=5.0, 
                           sigma_f=0.1):
    """
    This function calculates lower confidence bound
    """
    x_test = x_test 
    X_obs = X_obs 

    post_mean, post_cov = gp_post_par(x_test, X_obs, y_obs, sigma_y, ell, sigma_f)
    std = np.sqrt(np.diag(post_cov))
    
    lcb = -(post_mean - k*std)
    
    return lcb
```

|                                                              |
| :----------------------------------------------------------: |
| ![my image](images/projects/bayesian-optimization/lcb/0.png) |
| ![my image](images/projects/bayesian-optimization/lcb/1.png) |
| ![my image](images/projects/bayesian-optimization/lcb/2.png) |
| ![my image](images/projects/bayesian-optimization/lcb/3.png) |
| ![my image](images/projects/bayesian-optimization/lcb/4.png) |
| ![my image](images/projects/bayesian-optimization/lcb/5.png) |
| ![my image](images/projects/bayesian-optimization/lcb/6.png) |