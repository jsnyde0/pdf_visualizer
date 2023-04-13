import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm, uniform, expon, beta, gamma, lognorm, weibull_min

st.title('Interactive Probability Distribution Functions')

# Add a dropdown menu to select the distribution
distribution = st.sidebar.selectbox('Select distribution:', [
    'Binomial', 'Poisson', 'Normal (Gaussian)', 'Uniform', 'Exponential', 'Beta', 'Gamma', 'Log-normal', 'Weibull'
])

# Add sliders for each distribution's parameters and plot the distribution
if distribution == 'Binomial':
    n = st.sidebar.slider('Number of trials (n)', 1, 100, 10)
    p = st.sidebar.slider('Probability of success (p)', 0.0, 1.0, 0.5, step=0.01)

    x = np.arange(0, n+1)
    y = binom.pmf(x, n, p)

    plt.bar(x, y, alpha=0.5, color='mediumseagreen', label='Binomial Distribution')
    plt.xlabel('Number of successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distribution: n={n}, p={p}')
    plt.legend()

elif distribution == 'Poisson':
    rate = st.sidebar.slider('Average rate (λ)', 1.0, 10.0, 5.0, step=0.1)

    x = np.arange(0, 24)
    y = poisson.pmf(x, rate)

    plt.ylim(0.0, 0.4)
    plt.bar(x, y, alpha=0.5, color='mediumseagreen', label='Poisson Distribution')
    plt.xlabel('Number of events')
    plt.ylabel('Probability')
    plt.title(f'Poisson Distribution: λ={rate}')
    plt.legend()

elif distribution == 'Normal (Gaussian)':
    mu = st.sidebar.slider('Mean (μ)', -10.0, 10.0, 0.0, step=0.1)
    sigma = st.sidebar.slider('Standard deviation (σ)', 0.4, 5.0, 1.0, step=0.1)

    x = np.linspace(-10, 10, 1000)
    y = norm.pdf(x, mu, sigma)

    plt.ylim(0.0, 1.0)
    plt.plot(x, y, alpha=0.5, color='mediumseagreen', label='Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title(f'Normal Distribution: μ={mu}, σ={sigma}')
    plt.legend()

elif distribution == 'Uniform':
    a = st.sidebar.slider('Lower bound (a)', -10.0, 10.0, -5.0, step=0.1)
    b = st.sidebar.slider('Upper bound (b)', -10.0, 10.0, 5.0, step=0.1)

    if a < b:
        x = np.linspace(-10, 10, 1000)
        y = uniform.pdf(x, loc=a, scale=b-a)

        plt.ylim(0.0, 1.0)
        plt.plot(x, y, alpha=0.5, color='mediumseagreen', label='Uniform Distribution')
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.title(f'Uniform Distribution: a={a}, b={b}')
        plt.legend()
    else:
        st.sidebar.error('Lower bound (a) must be less than the upper bound (b).')

elif distribution == 'Exponential':
    rate = st.sidebar.slider('Rate (λ)', 0.1, 5.0, 1.0, step=0.1)

    x = np.linspace(0, 5, 1000)
    y = expon.pdf(x, scale=1/rate)

    plt.ylim(0.0, 3.0)
    plt.plot(x, y, alpha=0.5, color='mediumseagreen', label='Exponential Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title(f'Exponential Distribution: λ={rate}')
    plt.legend()

elif distribution == 'Beta':
    alpha_value = st.sidebar.slider('Alpha (α)', 0.1, 10.0, 1.0, step=0.1)
    beta_value = st.sidebar.slider('Beta (β)', 0.1, 10.0, 1.0, step=0.1)

    x = np.linspace(0, 1, 1000)
    y = beta.pdf(x, alpha_value, beta_value)

    plt.ylim(0.0, 5.0)
    plt.plot(x, y, alpha=0.5, color='mediumseagreen', label='Beta Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title(f'Beta Distribution: α={alpha_value}, β={beta_value}')
    plt.legend()

elif distribution == 'Gamma':
    shape = st.sidebar.slider('Shape (k)', 0.1, 10.0, 1.0, step=0.1)
    scale = st.sidebar.slider('Scale (θ)', 0.1, 10.0, 1.0, step=0.1)

    x = np.linspace(0, 3*shape*scale, 1000)
    y = gamma.pdf(x, shape, scale=scale)

    plt.plot(x, y, alpha=0.5, color='mediumseagreen', label='Gamma Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title(f'Gamma Distribution: k={shape}, θ={scale}')
    plt.legend()

elif distribution == 'Log-normal':
    mu = st.sidebar.slider('Mean (μ)', -10.0, 10.0, 0.0, step=0.1)
    sigma = st.sidebar.slider('Standard deviation (σ)', 0.1, 10.0, 1.0, step=0.1)

    x = np.linspace(0, np.exp(mu + 4*sigma), 1000)
    y = lognorm.pdf(x, sigma, scale=np.exp(mu))

    plt.plot(x, y, alpha=0.5, color='mediumseagreen', label='Log-normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title(f'Log-normal Distribution: μ={mu}, σ={sigma}')
    plt.legend()

elif distribution == 'Weibull':
    k = st.sidebar.slider('Shape (k)', 0.1, 10.0, 1.0, step=0.1)
    scale = st.sidebar.slider('Scale (λ)', 0.1, 10.0, 1.0, step=0.1)

    x = np.linspace(0, 3*scale, 1000)
    y = weibull_min.pdf(x, k, scale=scale)

    plt.plot(x, y, alpha=0.5, color='mediumseagreen', label='Weibull Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title(f'Weibull Distribution: k={k}, λ={scale}')
    plt.legend()

st.pyplot(plt.gcf())