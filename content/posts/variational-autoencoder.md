---
title: "Variational Autoencoders"
date: 2023-02-09T23:29:21+08:00
draft: false
ShowToc: true
category: [ai]
tags: ["ML", "AI", "Generative Modeling"]
category: ["ai"]
description: "Advanced Generative Modeling"
summary: "Advanced Generative Modeling"
mathjax: true
---


# Variational Autoencoders (VAEs) in Machine Learning

Here, we delve into the exploration of a prominent probabilistic model known as the Variational Autoencoder (VAE). VAEs are a deep learning technique designed for acquiring latent feature representations, and they have demonstrated significant prowess in various applications. These include image generation, achieving state-of-the-art performance in semi-supervised learning, and sentence interpolation.

## Deep Generative Models

To commence our discussion, let's consider a latent-variable model represented as:

$$ p(x,z) = p(x|z)p(z) $$

Here, $x$ represents observed data in a space $\mathcal{X}$, which can be continuous or discrete, and $z$ denotes latent variables residing in $\mathbb{R}^k$.

For instance, think of $x$ as an image (e.g., a human face), while $z$ represents latent factors, such as emotions (happy or sad), gender (male or female), etc. These latent variables aim to capture essential features of the observed data.

We might also encounter models with multiple layers, such as $p(x|z_1)p(z_1|z_2)p(z_2|z_3)\ldots p(z_{m-1}|z_m)p(z_m)$, known as *deep generative models*, which can learn hierarchical representations. For simplicity, we'll focus on single-layer models.

### Learning Deep Generative Models

Our objectives involve:

1. **Learning the parameters $\theta$ of $p$.**
2. **Approximating posterior inference over $z$**: Given an image $x$, what are its latent factors?
3. **Approximating marginal inference over $x$**: Given an image $x$ with missing parts, how can we impute these missing regions?

We operate under certain assumptions:

- **Intractability**: Computing the posterior probability $p(z|x)$ is computationally intractable.
- **Big Data**: The dataset $D$ is too large to fit in memory; we can only work with small, subsampled batches of $D$.

Many intriguing models fit this class, and one of the models we'll explore is the Variational Autoencoder.

## Traditional Approaches

Before introducing the Variational Autoencoder, let's briefly examine traditional techniques to address the tasks mentioned earlier.

- **Expectation-Maximization (EM)**: EM is commonly used for latent-variable models. However, it requires computing the approximate posterior $p(z|x)$, which we've assumed to be intractable. The M-step operates on the entire dataset, often exceeding memory capacity.

- **Mean Field**: Mean field methods are employed for approximate inference. But, for models where components of $x$ depend on all components of $z$, it becomes intractable due to the dependencies among $z$ variables.

- **Sampling-based methods**: Techniques like Metropolis-Hastings require hand-crafted proposal distributions and don't scale well to large datasets.

### Auto-encoding Variational Bayes (AEVB)

Now, let's explore the Auto-encoding Variational Bayes (AEVB) algorithm, which efficiently handles our inference and learning tasks, serving as the foundation for the Variational Autoencoder (VAE).

AEVB leverages ideas from variational inference, aiming to maximize the Evidence Lower Bound (ELBO):

$$
\mathcal{L}(p_{\theta},q_{\phi}) = \mathbb{E}q_{\phi(z|x)} \left[\log p_{\theta}(x,z) - \log q_{\phi}(z|x)\right]
$$

The ELBO satisfies:

$$
\log p_\theta(x) = KL(q_\phi(z|x) || p(z|x)) + \mathcal{L}(p_\theta,q_\phi)
$$

Here, $x$ is fixed, and $q(z|x)$ adapts to $x$ to improve the posterior approximation.

To optimize $q(z|x)$, we need a more versatile approach than mean field. This marks the introduction of *black-box variational inference*, enabling gradient descent optimization over $\phi$, assuming $q_\phi$ is differentiable in its parameters.

### The Score Function Gradient Estimator

In black-box variational inference, we compute the gradient:


$$\nabla_{\theta,\phi} \space \mathbb{E}q_{\phi(z)} \left[\log p_{\theta}(x,z) - \log q_{\phi}(z)\right]$$


However, computing the gradient with respect to $q$ is challenging since the expectation is taken over the distribution we are differentiating. The score function estimator comes to the rescue:

$$
\nabla_\phi \mathbb{E}q_{\phi(z)} \left[ \log p_\theta(x,z) - \log q_\phi(z) \right] = \mathbb{E}q_{\phi(z)} \left[ \left(\log p_\theta(x,z) - \log q_\phi(z) \right) \nabla_\phi \log q_\phi(z) \right]
$$

This estimator reduces the variance compared to other methods, enabling learning in complex models.

### The Reparametrization Trick

AEVB relies on the reparametrization trick, which reformulates $q_\phi(z|x)$ in two steps:

1. Sample a noise variable $\epsilon$ from a simple distribution $p(\epsilon)$.
2. Apply a deterministic transformation $g_\phi(\epsilon, x)$ to map the noise to a more complex distribution.

For instance, for Gaussian variables, the trick simplifies to $z=g_{\mu, \sigma}(\epsilon) = \mu + \epsilon \cdot \sigma$, where $\epsilon \sim \mathcal{N}(0,1)$. This low-variance estimator facilitates learning in models that were previously challenging.

### Choosing $q$ and $p$

To complete the AEVB setup, we need to specify $q$ and $p$. These distributions are parametrized using neural networks to ensure flexibility and expressiveness. For example, in a Gaussian setting:

$$q(z|x) = \mathcal{N}(z; \vec\mu(x), \text{diag}(\vec\sigma(x))^2)$$

The choice of $p$ and $q$ is pivotal in shaping the VAE's performance and capabilities.

## The Variational Autoencoder (VAE)

The Variational Autoencoder (VAE) is a specific instantiation of the AEVB algorithm. It employs a particular set of distributions and neural network structures.

For instance, it parametrizes $p$ as:

$$
\begin{align*}
p(x|z) & = \mathcal{N}(x; \vec\mu(z), \text{diag}(\vec\sigma(z))^2) \\\
p(z) & = \mathcal{N}(z; 0, I)
\end{align*}
$$

These parametrizations, achieved through neural networks, enable simplifications in the ELBO's computation, making it feasible to optimize.

In essence, the VAE aims to fit $q(z|x)$ to map $x$ into a useful latent space $z$ for reconstructing $x$ via $p(x|z)$, resembling the objective of autoencoder neural networks.

## Experimental Results

The VAE's utility becomes evident in experimental results. For instance, when applied to image datasets, the VAE can learn meaningful latent representations.

On the MNIST dataset, we can interpolate between numbers, creating smooth transitions. Additionally, the VAE has been compared to alternative approaches, such as the wake-sleep algorithm, Monte-Carlo EM, and hybrid Monte-Carlo. These comparisons illustrate the effectiveness of the VAE in various learning scenarios.

In summary, the Variational Autoencoder, built upon the foundations of the Auto-encoding Variational Bayes algorithm, represents a powerful tool in the realm of machine learning, offering rich opportunities for learning and generating latent representations in complex data spaces.
