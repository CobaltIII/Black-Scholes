import streamlit as st
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Stock price = S
# Strike price = K
# Time to maturity = T
# Volatility = Sigma
# Risk Free Interest Rate = r

st.set_page_config(page_title="Black-Scholes Pricing Model", layout="wide", initial_sidebar_state="expanded")

def N(x):
    return 0.5 * (1 + math.erf(x / (2 ** 0.5)))

def Call(S , K , T , Si, r):
    d1 = ((math.log(S / K)) + ((r + ((Si ** 2) / 2) * T))) / (Si * (T ** 0.5))
    d2 = d1 - (Si * (T ** 0.5))
    
    return (S * N(d1)) - (K * (math.e ** (-r * T)) * N(d2))

def Put(S , K , T , Si, r):
    d1 = ((math.log(S / K)) + ((r + ((Si ** 2) / 2) * T))) / (Si * (T ** 0.5))
    d2 = d1 - (Si * (T ** 0.5))
    
    return (K * (math.e ** (-r * T)) * N(-d2)) - (S * N(-d1))


#####################################################################

st.title("Black-Scholes Pricing Model")

st.sidebar.markdown("<h1 style='color:white;'>Black-Scholes Model</h1>", unsafe_allow_html=True)
st.sidebar.header("Made by Dhruv Jaiswal" , divider = "blue")
S = st.sidebar.number_input("Stock Price [$]", min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price [$]", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity [yrs]", min_value=0.01, value=1.0, step=0.01)
sigma = st.sidebar.number_input("Volatility", min_value=0.01, value=0.2, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", min_value=0.0, value=0.05, step=0.01)

st.sidebar.subheader("Heatmap Parameters")
min_S = st.sidebar.number_input("Min Spot Price", min_value=1.0, value=80.0, step=1.0)
max_S = st.sidebar.number_input("Max Spot Price", min_value=min_S + 1, value=120.0, step=1.0)
min_sigma = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.0, 0.1, 0.01)
max_sigma = st.sidebar.slider("Max Volatility for Heatmap", min_sigma, 1.0, 0.5, 0.01)

c = Call(S, K, T, sigma, r)
p = Put(S, K, T, sigma, r)

st.write("## Option Prices")


col1, col2 = st.columns(2)
col1.markdown(f"<div style='background-color:#90EE90; padding:20px; border-radius:10px; text-align:center; font-size:20px; font-weight:bold; color:#006400;'> CALL Value <br> <span style='font-size:24px;'>${c:.2f}</span> </div>", unsafe_allow_html=True)
col2.markdown(f"<div style='background-color:#FFB6C1; padding:20px; border-radius:10px; text-align:center; font-size:20px; font-weight:bold; color:#8B0000;'> PUT Value <br> <span style='font-size:24px;'>${p:.2f}</span> </div>", unsafe_allow_html=True)

#####################################################################

spot_prices = np.linspace(min_S, max_S, 10)
volatilities = np.linspace(min_sigma, max_sigma, 10)

call_prices = np.zeros((len(volatilities), len(spot_prices)))
put_prices = np.zeros((len(volatilities), len(spot_prices)))

for i, v in enumerate(volatilities):
    for j, s in enumerate(spot_prices):
        call_prices[i, j], put_prices[i, j] = Call(s, K, T, v, r), Put(s, K, T, v, r)
        if call_prices[i, j] < 10**(-2):
            call_prices[i, j] = 0
        if put_prices[i, j] < 10**(-2):
            put_prices[i, j] = 0
        if call_prices[i, j] > 10**(2):
            call_prices[i, j] = 100
        if put_prices[i, j] > 10**(2):
            put_prices[i, j] = 100

call_purchase_price = st.number_input("Purchase Price for Call", value=10.0)
put_purchase_price = st.number_input("Purchase Price for Put", value=10.0)

call_pnl = call_prices - call_purchase_price
put_pnl = put_prices - put_purchase_price

call_pnl_df = pd.DataFrame(call_pnl, index=np.round(volatilities, 2), columns=np.round(spot_prices, 2))
put_pnl_df = pd.DataFrame(put_pnl, index=np.round(volatilities, 2), columns=np.round(spot_prices, 2))

fig_pnl, axes_pnl = plt.subplots(1, 2, figsize=(14, 6))
fig_pnl.patch.set_facecolor('#0E1117')

for ax in axes_pnl:
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_color('white')

pnl_cmap = LinearSegmentedColormap.from_list("pnl_map", ["red", "black", "green"])

t11 = sns.heatmap(call_pnl_df, ax=axes_pnl[0], cmap=pnl_cmap, annot=True, center=0, fmt=".0f")
t22 = sns.heatmap(put_pnl_df, ax=axes_pnl[1], cmap=pnl_cmap, annot=True, center=0, fmt=".0f")

t11.collections[0].colorbar.ax.yaxis.set_tick_params(color='white')
t22.collections[0].colorbar.ax.yaxis.set_tick_params(color='white')

for tick_label in t11.collections[0].colorbar.ax.get_yticklabels():
    tick_label.set_color('white')
for tick_label in t22.collections[0].colorbar.ax.get_yticklabels():
    tick_label.set_color('white')

axes_pnl[0].set_title("Call PnL Heatmap", color='white')
axes_pnl[1].set_title("Put PnL Heatmap", color='white')
axes_pnl[0].set_xlabel("Spot Price", color='white')
axes_pnl[1].set_xlabel("Spot Price", color='white')
axes_pnl[0].set_ylabel("Volatility", color='white')
axes_pnl[1].set_ylabel("Volatility", color='white')

st.pyplot(fig_pnl)

#####################################################################

st.write("## Modelling with Heatmaps to show variation with volatility and spot price")

spot_prices = np.linspace(min_S, max_S, 10)
volatilities = np.linspace(min_sigma, max_sigma, 10)

call_prices = np.zeros((len(volatilities), len(spot_prices)))
put_prices = np.zeros((len(volatilities), len(spot_prices)))

for i, v in enumerate(volatilities):
    for j, s in enumerate(spot_prices):
        call_prices[i, j], put_prices[i, j] = Call(s, K, T, v, r), Put(s, K, T, v, r)
        if call_prices[i, j] < 10**(-2):
            call_prices[i, j] = 0
        if put_prices[i, j] < 10**(-2):
            put_prices[i, j] = 0
        if call_prices[i, j] > 10**(2):
            call_prices[i, j] = 100
        if put_prices[i, j] > 10**(2):
            put_prices[i, j] = 100

call_df = pd.DataFrame(call_prices, index=np.round(volatilities, 2), columns=np.round(spot_prices, 2))
put_df = pd.DataFrame(put_prices, index=np.round(volatilities, 2), columns=np.round(spot_prices, 2))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0E1117')

for ax in axes:
    ax.set_facecolor('#0E1117')

custom_cmap = LinearSegmentedColormap.from_list("custom", ["red", "black", "green"])
t1 = sns.heatmap(call_df, ax=axes[0], cmap=custom_cmap, annot=True, fmt=".1f")
t2 = sns.heatmap(put_df, ax=axes[1], cmap=custom_cmap, annot=True, fmt=".1f")

axes[0].set_title("Call Price Heatmap")
axes[1].set_title("Put Price Heatmap")
axes[0].set_xlabel("Spot Price")
axes[1].set_xlabel("Spot Price")
axes[0].set_ylabel("Volatility")
axes[1].set_ylabel("Volatility")

t1.collections[0].colorbar.ax.yaxis.set_tick_params(color='white')
t2.collections[0].colorbar.ax.yaxis.set_tick_params(color='white')

for tick_label in t1.collections[0].colorbar.ax.get_yticklabels():
    tick_label.set_color('white')
for tick_label in t2.collections[0].colorbar.ax.get_yticklabels():
    tick_label.set_color('white')

for ax in axes:
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")  

st.pyplot(fig)

#####################################################################

st.write("## Black Scholes Model")

st.markdown("""
### Key Variables
- $V(S,t)$: The price of an option as a function of stock price $S$ and time $t$.
- $C(S,t)$: The price of a European call option.
- $P(S,t)$: The price of a European put option.
- $T$: The time of option expiration.
- $\tau = T - t$: The time until maturity.
- $K$: The **strike price** (exercise price) of the option.

### Black-Scholes Partial Differential Equation

The Black-Scholes equation governing the price of an option is:
""")

st.latex(r"""
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
""")

st.markdown("""where:
- $\sigma$ is the volatility of the underlying asset.
- $r$ is the risk-free interest rate.

### Black-Scholes Formula for Call and Put Options
For a European **call** option:
""")

st.latex(r"""
C(S,t) = S N(d_1) - K e^{-r\tau} N(d_2)
""")

st.markdown("For a European **put** option:")

st.latex(r"""
P(S,t) = K e^{-r\tau} N(-d_2) - S N(-d_1)
""")

st.markdown("""
where:
""")

st.latex(r"""
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2} \sigma^2) \tau}{\sigma \sqrt{\tau}}
""")

st.latex(r"""
d_2 = d_1 - \sigma \sqrt{\tau}
""")

st.markdown("""
and $N(d)$ represents the cumulative standard normal distribution:
""")

st.latex(r"""
N(d) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{d} e^{-\frac{x^2}{2}} dx = \frac{1}{2} \left(1 + \text{erf}\left(\frac{d}{\sqrt{2}}\right)\right)
""")

st.markdown("""
This model provides a theoretical estimate for the fair price of options (assuming no arbitrage opportunities and a lognormal distribution of stock prices)
""")

#####################################################################

