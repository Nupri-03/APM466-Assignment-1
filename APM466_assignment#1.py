#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 28 19:33:33 2026

@author: nupri
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

bond_names = [
    'CAN 0.25 Mar 26',    # 2026 #1: ~0.15 years, 0.25%
    'CAN 1.00 Sep 26',    # 2026 #2: ~0.66 years, 1.00%
    'CAN 1.25 Mar 27',    # 2027 #1: ~1.15 years, 1.25%
    'CAN 2.75 Sep 27',    # 2027 #2: ~1.65 years, 2.75%
    'CAN 3.50 Mar 28',    # 2028 #1: ~2.15 years, 3.50%
    'CAN 3.25 Sep 28',    # 2028 #2: ~2.65 years, 3.25%
    'CAN 4.00 Mar 29',    # 2029 #1: ~3.15 years, 4.00%
    'CAN 3.50 Sep 29',    # 2029 #2: ~3.65 years, 3.50%
    'CAN 2.75 Mar 30',    # 2030 #1: ~4.15 years, 2.75%
    'CAN 2.75 Sep 30',    # 2030 #2: ~4.65 years, 2.75%
]

bond_data = {
    'CAN 0.25 Mar 26': {
        'coupon': 0.25, 'maturity': '2026-03-01', 'issue_date': '2020-10-09',
        'prices': [99.7, 99.71, 99.71, 99.72, 99.73, 99.74, 99.74, 99.75, 99.76, 99.77]
    },
    'CAN 1.00 Sep 26': {
        'coupon': 1.00, 'maturity': '2026-09-01', 'issue_date': '2021-04-16',
        'prices': [99.15, 99.15, 99.165, 99.16, 99.19, 99.18, 99.19, 99.20, 99.21, 99.22]
    },
    'CAN 1.25 Mar 27': {
        'coupon': 1.25, 'maturity': '2027-03-01', 'issue_date': '2021-10-15',
        'prices': [98.6, 98.63, 98.66, 98.67, 98.67, 98.67, 98.68, 98.67, 98.73, 98.72]
    },
    'CAN 2.75 Sep 27': {
        'coupon': 2.75, 'maturity': '2027-09-01', 'issue_date': '2022-05-13',
        'prices': [100.22, 100.30, 100.28, 100.31, 100.30, 100.32, 100.30, 100.31, 100.35, 100.37]
    },
    'CAN 3.50 Mar 28': {
        'coupon': 3.50, 'maturity': '2028-03-01', 'issue_date': '2022-10-21',
        'prices': [101.73, 101.78, 101.78, 101.80, 101.79, 101.81, 101.78, 101.81, 101.84, 101.83]
    },
    'CAN 3.25 Sep 28': {
        'coupon': 3.25, 'maturity': '2028-09-01', 'issue_date': '2023-04-21',
        'prices': [101.34, 101.41, 101.40, 101.43, 101.42, 101.45, 101.42, 101.43, 101.48, 101.47]
    },
    'CAN 4.00 Mar 29': {
        'coupon': 4.00, 'maturity': '2029-03-01', 'issue_date': '2023-10-13',
        'prices': [103.63, 103.70, 103.71, 103.74, 103.73, 103.76, 103.72, 103.76, 103.79, 103.78]
    },
    'CAN 3.50 Sep 29': {
        'coupon': 3.50, 'maturity': '2029-09-01', 'issue_date': '2024-04-08',
        'prices': [102.22, 102.33, 102.37, 102.34, 102.31, 102.35, 102.29, 102.33, 102.43, 102.42]
    },
    'CAN 2.75 Mar 30': {
        'coupon': 2.75, 'maturity': '2030-03-01', 'issue_date': '2024-10-03',
        'prices': [99.49, 99.42, 99.56, 99.50, 99.58, 99.528, 99.503, 99.658, 99.663, 99.613]
    },
    'CAN 2.75 Sep 30': {
        'coupon': 2.75, 'maturity': '2030-09-01', 'issue_date': '2025-04-10',
        'prices': [99.17, 99.09, 99.25, 99.17, 99.26, 99.21, 99.185, 99.355, 99.365, 99.315]
    },
}

# Date setup
dates = ['Jan 5', 'Jan 6', 'Jan 7', 'Jan 8', 'Jan 9', 'Jan 12', 'Jan 13', 'Jan 14', 'Jan 15', 'Jan 16']
date_objects = [
    datetime(2026, 1, 5),
    datetime(2026, 1, 6),
    datetime(2026, 1, 7),
    datetime(2026, 1, 8),
    datetime(2026, 1, 9),
    datetime(2026, 1, 12),
    datetime(2026, 1, 13),
    datetime(2026, 1, 14),
    datetime(2026, 1, 15),
    datetime(2026, 1, 16),
]
reference_date = datetime(2026, 1, 5)


# accrued interest


def get_last_coupon_date(maturity_date_str, settlement_date):
    maturity = datetime.strptime(maturity_date_str, '%Y-%m-%d')
    
    # Work backwards from maturity in 6-month intervals
    coupon_date = maturity
    while coupon_date > settlement_date:
        coupon_date = coupon_date - relativedelta(months=6)
    
    return coupon_date

def get_next_coupon_date(maturity_date_str, settlement_date):

    maturity = datetime.strptime(maturity_date_str, '%Y-%m-%d')
    
    # Work backwards from maturity to find the bracket
    coupon_date = maturity
    while coupon_date > settlement_date:
        prev_coupon = coupon_date
        coupon_date = coupon_date - relativedelta(months=6)
    
    return prev_coupon

def calculate_accrued_interest(coupon, maturity_date_str, settlement_date):

    last_coupon = get_last_coupon_date(maturity_date_str, settlement_date)
    next_coupon = get_next_coupon_date(maturity_date_str, settlement_date)
    
    days_accrued = (settlement_date - last_coupon).days
    days_in_period = (next_coupon - last_coupon).days
    
    # Semi-annual coupon payment
    semi_annual_coupon = coupon / 2
    
    accrued = semi_annual_coupon * (days_accrued / days_in_period)
    
    return accrued


def calculate_ttm(maturity_date, ref_date=reference_date):

    mat = datetime.strptime(maturity_date, '%Y-%m-%d')
    diff_days = (mat - ref_date).days
    return diff_days / 365.0

def calculate_ytm(dirty_price, coupon, ttm, face_value=100):

    #Calculate yield to maturity using Newton-Raphson method.

    periods = int(np.ceil(ttm * 2))
    coupon_payment = coupon / 2
    
    ytm = coupon / 100  
    tolerance = 1e-6
    max_iter = 1000
    
    for _ in range(max_iter):
        pv = 0
        pv_derivative = 0
        
        for j in range(1, periods + 1):
            t = j * 0.5
            discount = np.exp(-ytm * t)
            
            if j == periods:
                cash_flow = coupon_payment + face_value
            else:
                cash_flow = coupon_payment
            
            pv += cash_flow * discount
            pv_derivative += -cash_flow * t * discount
        
        diff = pv - dirty_price
        if abs(diff) < tolerance:
            return ytm * 100
        
        ytm = ytm - diff / pv_derivative
    
    return ytm * 100

def bootstrap_spot_rates(bond_names, bond_data, dirty_prices_for_day):

    n_bonds = len(bond_names)
    spot_rates = np.zeros(n_bonds)
    ttms = np.zeros(n_bonds)
    
    # First bond - spot rate equals YTM
    bond_0 = bond_names[0]
    dirty_price_0 = dirty_prices_for_day[0]
    coupon_0 = bond_data[bond_0]['coupon']
    ttm_0 = calculate_ttm(bond_data[bond_0]['maturity'])
    ttms[0] = ttm_0
    spot_rates[0] = calculate_ytm(dirty_price_0, coupon_0, ttm_0) / 100
    
    for i in range(1, n_bonds):
        bond_i = bond_names[i]
        dirty_price_i = dirty_prices_for_day[i]
        coupon_i = bond_data[bond_i]['coupon'] / 2  
        ttm_i = calculate_ttm(bond_data[bond_i]['maturity'])
        ttms[i] = ttm_i
        periods_i = int(np.ceil(ttm_i * 2))
        
        pv_coupons = 0
        for j in range(1, periods_i):
            t_j = j * 0.5
            if t_j <= ttms[i-1]:
                spot_j = np.interp(t_j, ttms[:i], spot_rates[:i])
            else:
                spot_j = spot_rates[i-1]
            
            pv_coupons += coupon_i * np.exp(-spot_j * t_j)
        
        final_payment = coupon_i + 100
        remaining_pv = dirty_price_i - pv_coupons
        
        if remaining_pv > 0:
            spot_rates[i] = np.log(final_payment / remaining_pv) / ttm_i
        else:
            spot_rates[i] = spot_rates[i-1]  # Fallback
    
    return spot_rates * 100, ttms

def calculate_forward_rates(spot_rates, ttms):
    t1 = 1.0
    forward_rates = []
    
    spot_t1 = np.interp(t1, ttms, spot_rates) / 100
    
    for t2 in range(2, 6):
        if t2 <= ttms[-1]:
            spot_t2 = np.interp(t2, ttms, spot_rates) / 100
        else:
            slope = (spot_rates[-1] - spot_rates[-2]) / (ttms[-1] - ttms[-2])
            spot_t2 = (spot_rates[-1] + slope * (t2 - ttms[-1])) / 100
        
        forward_rate = (spot_t2 * t2 - spot_t1 * t1) / (t2 - t1)
        forward_rates.append(forward_rate * 100)
    
    return forward_rates

#calculate dirty prices

print("=" * 80)
print("CALCULATING DIRTY PRICES (Clean Price + Accrued Interest)")
print("=" * 80)

dirty_prices = {}
for bond in bond_names:
    dirty_prices[bond] = []
    coupon = bond_data[bond]['coupon']
    maturity = bond_data[bond]['maturity']
    clean_prices = bond_data[bond]['prices']
    
    print(f"\n{bond}:")
    for day_idx, settlement_date in enumerate(date_objects):
        clean_price = clean_prices[day_idx]
        accrued = calculate_accrued_interest(coupon, maturity, settlement_date)
        dirty_price = clean_price + accrued
        dirty_prices[bond].append(dirty_price)
        
        print(f"  {dates[day_idx]}: Clean={clean_price:.4f}, Accrued={accrued:.4f}, Dirty={dirty_price:.4f}")

print("\n" + "=" * 80)
print("SELECTED 10 BONDS FOR 0-5 YEAR YIELD AND SPOT CURVES")
print("=" * 80)
for i, bond in enumerate(bond_names):
    ttm = calculate_ttm(bond_data[bond]['maturity'])
    print(f"{i+1}. {bond:20s} | TTM: {ttm:5.2f} years | Coupon: {bond_data[bond]['coupon']:5.2f}%")
print()

#calculate ytm, spot, forward

ytm_matrix = np.zeros((len(bond_names), 10))
spot_matrix = np.zeros((len(bond_names), 10))
forward_matrix = np.zeros((4, 10))

for day_idx in range(10):
    dirty_prices_day = [dirty_prices[bond][day_idx] for bond in bond_names]
    
    for bond_idx, bond in enumerate(bond_names):
        coupon = bond_data[bond]['coupon']
        ttm = calculate_ttm(bond_data[bond]['maturity'])
        ytm = calculate_ytm(dirty_prices_day[bond_idx], coupon, ttm)
        ytm_matrix[bond_idx, day_idx] = ytm
    
    spot_rates, ttms = bootstrap_spot_rates(bond_names, bond_data, dirty_prices_day)
    spot_matrix[:, day_idx] = spot_rates
    
    forward_rates = calculate_forward_rates(spot_rates, ttms)
    forward_matrix[:, day_idx] = forward_rates


# Q4(a): Plot Yield Curves
plt.figure(figsize=(12, 7))
X_ytm = np.linspace(0, 5, 50)

for day_idx in range(10):
    ytms_day = ytm_matrix[:, day_idx]
    ttms_for_interp = [calculate_ttm(bond_data[bond]['maturity']) for bond in bond_names]
    cs = CubicSpline(ttms_for_interp, ytms_day, bc_type='natural')
    ytm_interpolated = cs(X_ytm)
    plt.plot(X_ytm, ytm_interpolated, label=dates[day_idx], alpha=0.7)

plt.title('5-Year Yield Curves (Jan 5-16, 2026) - DIRTY PRICES', fontsize=14, fontweight='bold')
plt.xlabel('Time to Maturity (Years)', fontsize=12)
plt.ylabel('Yield to Maturity (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig('yield_curves_dirty.png', dpi=300)
plt.show()

# Q4(b): Plot Spot Curves
plt.figure(figsize=(12, 7))

for day_idx in range(10):
    spot_rates_day = spot_matrix[:, day_idx]
    ttms_for_plot = [calculate_ttm(bond_data[bond]['maturity']) for bond in bond_names]
    plt.plot(ttms_for_plot, spot_rates_day, marker='o', label=dates[day_idx], alpha=0.7)

plt.title('5-Year Spot Rate Curves (Jan 5-16, 2026) - DIRTY PRICES', fontsize=14, fontweight='bold')
plt.xlabel('Time to Maturity (Years)', fontsize=12)
plt.ylabel('Spot Rate (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig('spot_curves_dirty.png', dpi=300)
plt.show()

# Q4(c): Plot Forward Curves
plt.figure(figsize=(12, 7))
forward_labels = ['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr']

for day_idx in range(10):
    plt.plot(range(4), forward_matrix[:, day_idx], marker='^', label=dates[day_idx], alpha=0.7)

plt.title('1-Year Forward Rate Curves (Jan 5-16, 2026) - DIRTY PRICES', fontsize=14, fontweight='bold')
plt.xlabel('Forward Rate Tenor', fontsize=12)
plt.ylabel('Forward Rate (%)', fontsize=12)
plt.xticks(range(4), forward_labels)
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig('forward_curves_dirty.png', dpi=300)
plt.show()

#covariance matrices

target_maturities = [1, 2, 3, 4, 5]
num_days = 10
yield_5_rates = np.zeros((5, num_days))

for day_idx in range(10):
    ttms = np.array([calculate_ttm(bond_data[bond]['maturity']) for bond in bond_names])
    ytms = ytm_matrix[:, day_idx]
    order = np.argsort(ttms)
    ttms_sorted = ttms[order]
    ytms_sorted = ytms[order]
    
    for i, T in enumerate(target_maturities):
        yield_5_rates[i, day_idx] = np.interp(T, ttms_sorted, ytms_sorted)

print("YIELDS AT 1–5 YEAR MATURITIES (DIRTY PRICES)")
print(yield_5_rates)
print()

yield_log_returns = np.log(yield_5_rates[:, 1:] / yield_5_rates[:, :-1])
forward_matrix = forward_matrix[0:4, :]
forward_log_returns = np.log(forward_matrix[:, 1:] / forward_matrix[:, :-1])

cov_yield = np.cov(yield_log_returns, rowvar=True)
cov_forward = np.cov(forward_log_returns, rowvar=True)

print("COVARIANCE MATRIX - YIELDS (DIRTY PRICES)")
print(cov_yield)
print()

print("COVARIANCE MATRIX - FORWARDS (DIRTY PRICES)")
print(cov_forward)
print()

# Eigendecomposition
eigvals_yield, eigvecs_yield = np.linalg.eigh(cov_yield)
eigvals_forward, eigvecs_forward = np.linalg.eigh(cov_forward)

idx = np.argsort(eigvals_yield)[::-1]
eigvals_yield = eigvals_yield[idx]
eigvecs_yield = eigvecs_yield[:, idx]

idx = np.argsort(eigvals_forward)[::-1]
eigvals_forward = eigvals_forward[idx]
eigvecs_forward = eigvecs_forward[:, idx]

print("EIGENVALUES - YIELDS (DIRTY PRICES)")
print(eigvals_yield)
print()

print("EIGENVECTORS - YIELDS (DIRTY PRICES)")
print(eigvecs_yield)
print()

print("EIGENVALUES - FORWARDS (DIRTY PRICES)")
print(eigvals_forward)
print()

print("EIGENVECTORS - FORWARDS (DIRTY PRICES)")
print(eigvecs_forward)
print()