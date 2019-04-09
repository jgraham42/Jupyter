# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 07:53:52 2017

@author: jgraham
"""

from scipy.stats import beta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Facebook sign up vs conventional user/password sign up for Spotify
# Calculate priors using a sample of 100 users
users = 50

# Our control is conventional sign up, the experiment is Facebook sign up
control, experiment = np.random.rand(2,users)

c_successes = sum(control < .10)

# Let's say that Facebook sign up leads to a 26.6% conversion rate
e_successes = sum(experiment < .266)

c_failures = users - c_successes
e_failures = users - e_successes

# Priors (based on the intial read that conventional sign up leads to a 10% conv rate)
prior_successes = 5
prior_failures = 45

fig, ax = plt.subplots(1,1)

# Control 
c_alpha, c_beta = c_successes + prior_successes, c_failures + prior_failures

# Experiment
e_alpha, e_beta = e_successes + prior_successes, e_failures + prior_failures

x = np.linspace(0., 0.5, 1000)

# Generate and plot the distribution
c_distribution = beta(c_alpha, c_beta)
e_distribution = beta(e_alpha, e_beta)

# Plot our Priors
ax.plot(x,c_distribution.pdf(x), 'k-', label = 'Conv Sign Up')
ax.plot(x,e_distribution.pdf(x), 'c-', label = 'Facebook Sign Up')
ax.legend()
ax.set(xlabel = 'Conversion Rate', ylabel = 'Density')
fig

# We need a larger sample
more_users = 10000
control, experiment = np.random.rand(2, more_users)
c_successes += sum(control < .15)
e_successes += sum(experiment < .266)
c_failures += more_users - sum(control < .15)
e_failures += more_users - sum(experiment < .266)

# Plot the larger Sample
fig, ax = plt.subplots(1,1)

# Control 
c_alpha, c_beta = c_successes + prior_successes, c_failures + prior_failures

# Experiment
e_alpha, e_beta = e_successes + prior_successes, e_failures + prior_failures

x = np.linspace(0., 0.5, 1000)

# Generate and plot the distribution
c_distribution = beta(c_alpha, c_beta)
e_distribution = beta(e_alpha, e_beta)

# Plot our Priors
ax.plot(x,c_distribution.pdf(x), 'k-', label = 'Conv Sign Up')
ax.plot(x,e_distribution.pdf(x), 'c-', label = 'Facebook Sign Up')
ax.legend()
ax.set(xlabel = 'Conversion Rate', ylabel = 'Density')
fig