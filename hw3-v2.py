# %% [markdown]
# # Q2

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.stats import norm
from scipy.optimize import root_scalar

# ============================================================================
# HELPER FUNCTIONS FROM TEMPLATE
# ============================================================================

def HestonFourierPrice(T, F, D, v0, vBar, lambda_, eta, rho, payoffFourierTransform, beta):
    """
    Compute Heston price of a European contingent claim via Fourier transform
    """
    f = np.log(F)
    rhoEta = rho * eta
    eta2 = eta ** 2
    lambdaOverEta2 = lambda_ / eta2
    
    def CharacteristicFunctionPrice(z):
        tmp1 = lambda_ - rhoEta * z
        d = np.sqrt(tmp1 ** 2 - eta2 * z * (z - 1))
        tmp2 = tmp1 - d
        tmp3 = np.exp(-d * T)
        g = tmp2 / (tmp1 + d)
        tmp4 = g * tmp3 - 1
        C = lambdaOverEta2 * (tmp2 * T - 2 * np.log(tmp4 / (g - 1)))
        D = (tmp2 / eta2) * ((tmp3 - 1) / tmp4)
        return np.exp(z * f + C * vBar + D * v0)
    
    def Integrand(omega):
        return np.real(payoffFourierTransform(omega - 1j * beta) * 
                      CharacteristicFunctionPrice(beta + 1j * omega))
    
    return D * quad(Integrand, 0, np.inf, full_output=1)[0] / np.pi


def CallPayoffFourierTransform(K):
    """
    Generates the Fourier transform of a European call option payoff with strike K
    """
    k = np.log(K)
    def Ghat(omega):
        iomega = 1j * omega
        return np.exp((1 - iomega) * k) / (iomega * (iomega - 1))
    return Ghat


def bs_call_price(S0, K, T, r, sigma):
    """Black-Scholes call option price"""
    if sigma <= 0 or T <= 0:
        return max(S0 - K, 0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_call(C, S0, K, T, r):
    """Calculate implied volatility from call price"""
    intrinsic = max(S0 - K * np.exp(-r * T), 0)
    if C <= intrinsic + 1e-8:
        return np.nan
    try:
        sol = root_scalar(lambda sig: bs_call_price(S0, K, T, r, sig) - C,
                         bracket=[1e-4, 5.0], method='brentq')
        return sol.root if sol.converged else np.nan
    except:
        return np.nan


def estimateDiscountFactor(row):
    """Estimate discount factor from put-call parity"""
    avgK = row['Strike Price'].mean()
    avgO = (row['Call Premium'] - row['Put Premium']).mean()
    avgKK = (row['Strike Price'] ** 2).mean()
    avgKO = (row['Strike Price'] * (row['Call Premium'] - row['Put Premium'])).mean()
    return (avgKO - avgK * avgO) / (avgK ** 2 - avgKK)


def estimateForwardPrice(row):
    """Estimate forward price from put-call parity"""
    avgK = row['Strike Price'].mean()
    avgO = (row['Call Premium'] - row['Put Premium']).mean()
    avgKK = (row['Strike Price'] ** 2).mean()
    avgKO = (row['Strike Price'] * (row['Call Premium'] - row['Put Premium'])).mean()
    return (avgK * avgKO - avgKK * avgO) / (avgKO - avgK * avgO)


# %%
# ============================================================================
# QUESTION 2: HESTON CALIBRATION
# ============================================================================

print("="*80)
print("QUESTION 2: HESTON MODEL CALIBRATION")
print("="*80)

data = pd.read_csv('Midprices.csv')
target_data = data[(data['As of Date'] == '8/7/2024') & 
                    (data['Expiration Date'] == '9/6/2024')]

target_strikes = [5105, 5155, 5205, 5255, 5305]
market_data = target_data[target_data['Strike Price'].isin(target_strikes)].copy()
market_data = market_data.sort_values('Strike Price')

market_call_prices = market_data['Call Premium'].values
strikes = market_data['Strike Price'].values
S0 = market_data['Underlying Price'].iloc[0]
T = market_data['Time to Expiration'].iloc[0]

D = estimateDiscountFactor(target_data)
F = estimateForwardPrice(target_data)

print(f"\nData: S0={S0:.2f}, F={F:.2f}, D={D:.6f}, T={T:.4f}")

v0, vBar, lambda_ = 0.08364961, 0.05127939, 1.697994

def objective(params):
    eta, rho = params
    if eta <= 0 or rho < -1 or rho > 1:
        return 1e10
    try:
        model_prices = [HestonFourierPrice(T, F, D, v0, vBar, lambda_, eta, rho,
                                            CallPayoffFourierTransform(K), beta=1.5)
                        for K in strikes]
        return np.sum((market_call_prices - np.array(model_prices))**2)
    except:
        return 1e10

print("\nCalibrating...")
result = differential_evolution(objective, bounds=[(0.0001, 20), (-1, 1)],
                                seed=0, polish=True, maxiter=10000)

eta_cal, rho_cal = result.x
print(f"\n*** CALIBRATED PARAMETERS ***")
print(f"η = {eta_cal:.6f}")
print(f"ρ = {rho_cal:.6f}")

feller = 2 * lambda_ * vBar
print(f"\n*** FELLER'S CONDITION ***")
print(f"2λv̄ = {feller:.6f},  η² = {eta_cal**2:.6f}")
print(f"Satisfied: {feller >= eta_cal**2}")

print(f"\n*** FIT QUALITY ***")
for i, K in enumerate(target_strikes):
    mp = HestonFourierPrice(T, F, D, v0, vBar, lambda_, eta_cal, rho_cal,
                            CallPayoffFourierTransform(K), beta=1.5)
    print(f"K={K}: Market={market_call_prices[i]:.2f}, Model={mp:.2f}")


# %%
"""
HOMEWORK 3 - QUESTION 3: POWER PUT PRICING (CORRECTED)

Using the formula from Q1 solution:
Price = D * (K^n / 2π) * Σ C(n,j)(-1)^j * ∫ [j-(β+iω)]^-1 * E[(S_T/K)^(β+iω)] dω

where E[(S_T/K)^z] = K^-z * E[S_T^z] is computed via Heston characteristic function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.special import comb

# ============================================================================
# QUESTION 3: POWER PUT PRICING 
# ============================================================================

def PowerPutPayoffFourierTransform(K, n):
    """
    Fourier transform of power put payoff [(K - e^x)^+]^n
    
    From Q4 solution:
    Ĝ_n(ω) = K^(n-iω) * Σ_{j=0}^n C(n,j) * (-1)^j / (j - iω)
    
    Valid for Im(ω) > 0
    
    This is the KEY insight: we can directly use this as payoffFourierTransform!
    """
    k = np.log(K)
    
    def Ghat(omega):
        """
        Compute Ĝ_n(ω) using the formula from Q4 solution.
        """
        iomega = 1j * omega
        
        # K^(n-iω) * Σ C(n,j) * (-1)^j / (j - iω)
        result = K**(n - iomega)
        
        sum_term = 0.0
        for j in range(n + 1):
            binomial_coeff = comb(n, j, exact=True)
            sum_term += binomial_coeff * ((-1)**j) / (j - iomega)
        
        return result * sum_term
    
    return Ghat

K_range = np.arange(100, 12001, 100)
powers = [1, 2, 3]
results = {}

# From Q1: Ĝ_n(ω) is valid for Im(ω) > 0
# From Q1: Use β < 0 to ensure Ĝ_n(ω - iβ) is defined for all real ω
#
# Analysis: For ω real and β real:
#   ω - iβ has imaginary part = Im(ω - iβ) = -β
#   We need -β > 0, so β < 0
#
beta = -0.5  # NEGATIVE beta as required by Q5 solution

for n in powers:
    print(f"\nComputing n = {n} (using β = {beta})...")
    prices = []
    valid_strikes = []
    
    for i, K in enumerate(K_range):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(K_range)}")
        
        try:
            price = HestonFourierPrice(T, F, D, v0, vBar, lambda_, eta_cal, rho_cal,
                                        PowerPutPayoffFourierTransform(K, n), 
                                        beta=beta)
            
            if price >= 0:
                prices.append(price)
                valid_strikes.append(K)
        except Exception as e:
            print(f"    Error at K={K}: {e}")
            continue
    
    results[n] = {'strikes': np.array(valid_strikes), 
                    'prices': np.array(prices)}
    
    print(f"Computed {len(prices)} prices")
    if len(prices) >= 5:
        indices = [0, len(prices)//4, len(prices)//2, 3*len(prices)//4, -1]
        for idx in indices:
            print(f"  K={valid_strikes[idx]:6.0f}: {prices[idx]:.6e}")

# Verify monotonicity
print("\n" + "="*80)
print("MONOTONICITY CHECK")
print("="*80)
for n in powers:
    prices = results[n]['prices']
    if len(prices) > 1:
        diffs = np.diff(prices)
        num_decreases = np.sum(diffs < -1e-6)
        print(f"n={n}: Monotonically increasing? {num_decreases == 0}")
        if num_decreases > 0:
            print(f"  (Note: {num_decreases} decrease(s))")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['blue', 'green', 'red']

for idx, n in enumerate(powers):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    if len(results[n]['strikes']) > 0:
        ax.plot(results[n]['strikes'], results[n]['prices'],
                color=colors[idx], linewidth=2)
        ax.axvline(F, color='black', linestyle='--', alpha=0.5,
                    label=f'Forward = {F:.0f}')
        ax.set_xlabel('Strike K')
        ax.set_ylabel('Power Put Price')
        ax.set_title(f'n = {n}')
        ax.legend()
        ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for idx, n in enumerate(powers):
    if len(results[n]['strikes']) > 0:
        ax.semilogy(results[n]['strikes'], results[n]['prices'],
                    color=colors[idx], linewidth=2, label=f'n = {n}')
ax.axvline(F, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Strike K')
ax.set_ylabel('Price (log scale)')
ax.set_title('All Powers')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save CSV
max_len = max(len(results[n]['strikes']) for n in powers)
df_data = {'Strike': []}
for n in powers:
    df_data[f'Price_n{n}'] = []

for i in range(max_len):
    strike = None
    for n in powers:
        if i < len(results[n]['strikes']):
            if strike is None:
                strike = results[n]['strikes'][i]
    
    if strike is not None:
        df_data['Strike'].append(strike)
        for n in powers:
            if i < len(results[n]['strikes']) and results[n]['strikes'][i] == strike:
                df_data[f'Price_n{n}'].append(results[n]['prices'][i])
            else:
                df_data[f'Price_n{n}'].append(np.nan)

df = pd.DataFrame(df_data)
df.to_csv('q3_power_put.csv', index=False)
print("Saved: q3_power_put.csv")


# %% [markdown]
# # Q4


