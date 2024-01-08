import numpy as np
import math
#from scipy.stats import norm


def _get_option_delta(s, k, r, t, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    delta = norm.cdf(d1)
    return delta

def _get_option_price(s, k, r, t, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    option_price = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    return option_price

def norm_cdf(x, mu=0, sigma=1):
    #return (1 + np.erf((x - mu) / (sigma * np.sqrt(2)))) / 2
    return (1 + math.erf(x / math.sqrt(2))) / 2 

def get_option_delta(s, k, r, t, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    #delta = norm.cdf(d1)
    delta = norm_cdf(d1)
    return delta

def get_option_price(s, k, r, t, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    #option_price = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    option_price = s * norm_cdf(d1) - k * np.exp(-r * t) * norm_cdf(d2)
    return option_price


def test_get_option_delta():
    assert np.isclose(get_option_delta(100, 100, 0.05, 1, 0.2), 0.6368306511756191, rtol=1e-5)

def test_get_option_price():
    assert np.isclose(get_option_price(100, 100, 0.05, 1, 0.2), 10.4506, rtol=1e-4)

test_get_option_delta()
test_get_option_price()

from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta


def test_get_option_delta():
    s = 100
    k = 100
    r = 0.05
    t = 1
    sigma = 0.2
    expected_delta = delta('c', s, k, t, r, sigma)
    assert np.isclose(get_option_delta(s, k, r, t, sigma), expected_delta, rtol=1e-5)

def test_get_option_price():
    s = 100
    k = 100
    r = 0.05
    t = 1
    sigma = 0.2
    expected_price = black_scholes('c', s, k, t, r, sigma)
    assert np.isclose(get_option_price(s, k, r, t, sigma), expected_price, rtol=1e-4)


test_get_option_delta()
test_get_option_price()

