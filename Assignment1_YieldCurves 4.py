#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from datetime import datetime
from math import *
from scipy import optimize
import datetime
from scipy.interpolate import make_interp_spline, BSpline
import scipy as sp
from numpy import linalg as LA


# In[33]:


bonds = pd.read_excel('./data/selected_bonds_2022-01-13.xlsx', sheet_name='Sheet3', parse_dates=['Issue Date', 'Maturity Date'])
print(bonds)


# In[34]:


close = pd.read_excel('./data/selected_close_2022-01-13.xlsx', parse_dates=['Date'], sheet_name='Sheet3')
print(close)


# In[35]:


tr, yr = [], []
today = datetime.datetime(2022, 1, 10)
bonds['ttm'] = [(maturity - today).days / 365 for maturity in bonds['Maturity Date']]
bonds = bonds.sort_values(by=['ttm'])
print(bonds)

for i, bond in bonds.iterrows():
    par = 100
    ttm = bond['ttm']
    coupon = par * bond['Coupon'] / 2. # semi-annual coupon payment
    price = (0.5 - ttm % 0.5) * bond['Coupon'] + close.iloc[close.loc[close['Date'] == today].index[0], i+1]
    def RHS_func(ytm): # ttm, coupon, par, PRICE
        if ttm <= (ttm % 0.5): # zero-coupon bond maturing soon
            return (coupon + par) * np.exp(- ttm * ytm) - price
        else: # non-zero-coupon bonds
            temp = 0
            for j in np.arange(ttm, 0, -0.5):
                temp += coupon * np.exp(- j * ytm)
            return temp + par * np.exp(- ttm * ytm) - price
    ytm = optimize.newton(RHS_func, 0.01)
    print('Bond ' + str(i) + ': ' + str(ytm))
    tr.append(ttm)
    yr.append(ytm)

bonds['ytm'] = yr
print(bonds)
xlabel('Time to maturity'), ylabel('Yield to maturity'), grid(True)
plot(tr, np.array(yr)*100, marker='^', label=today.strftime("%m/%d/%Y"))
legend(loc='upper right'), show()


# In[36]:


tr, yr = [], []
today = datetime.datetime(2022, 1, 10)
bonds['ttm'] = [(maturity - today).days / 365 for maturity in bonds['Maturity Date']]
bonds = bonds.sort_values(by=['ttm'])
print(bonds)

for i, bond in bonds.iterrows():
    par = 100
    ttm = bond['ttm']
    coupon = par * bond['Coupon'] / 2. # semi-annual coupon payment
    price = (0.5 - ttm % 0.5) * bond['Coupon'] + close.iloc[close.loc[close['Date'] == today].index[0], i+1]
    def RHS_func(ytm): # ttm, coupon, par, PRICE
      if ttm <= (ttm % 0.5): # zero-coupon bond maturing soon
        return (coupon + par) * np.exp(- ttm * ytm) - price
      else: # non-zero-coupon bonds
        temp = 0
        for j in np.arange(ttm, 0, -0.5):
          temp += coupon * np.exp(- j * ytm)
        return temp + par * np.exp(- ttm * ytm) - price
    ytm = optimize.newton(RHS_func, 0.01)
    print('Bond ' + str(i) + ': ' + str(ytm))
    tr.append(ttm)
    yr.append(ytm)

bonds['ytm'] = yr
print(bonds)
xlabel('Time to maturity'), ylabel('Yield to maturity'), grid(True)
plot(tr, np.array(yr)*100, marker='^', label=today.strftime("%m/%d/%Y"))
legend(loc='upper right'), show()


# In[37]:


ts = np.linspace(min(tr), max(tr), 300)
spl = make_interp_spline(tr, yr, k=3)
ys = spl(ts)

xlabel('Time to maturity'), ylabel('Yield to maturity'), grid(True)
plt.plot(ts, ys, label=today.strftime("%m/%d/%Y"))
plt.scatter(tr, yr, marker='x', color='red')
legend(loc='upper right')
plt.show()


# #### Superimposed yield curve

# In[38]:


for today in close['Date']:
  tr, yr = [], []
  # today = datetime.datetime(2022, 1, 10)
  bonds['ttm'] = [(maturity - today).days / 365 for maturity in bonds['Maturity Date']]
  bonds = bonds.sort_values(by=['ttm'])
  # print(bonds)


  for i, bond in bonds.iterrows():
      par = 100
      ttm = bond['ttm']
      coupon = par * bond['Coupon']/2 # semi-annual coupon payment
      price = (0.5 - ttm % 0.5) * bond['Coupon'] + close.iloc[close.loc[close['Date'] == today].index[0], i+1]
      def RHS_func(ytm): # ttm, coupon, par, PRICE
        if ttm <= (ttm % 0.5): # zero-coupon bond maturing soon
          return (coupon + par) * np.exp(- ttm * ytm) - price
        else: # non-zero-coupon bonds
          temp = 0
          for j in np.arange(ttm, 0, -0.5):
            temp += coupon * np.exp(- j * ytm)
          return temp + par * np.exp(- ttm * ytm) - price
      ytm = optimize.newton(RHS_func, 0.01)
      # print('Bond ' + str(i) + ': ' + str(ytm))
      tr.append(ttm)
      yr.append(ytm)


  ts = np.linspace(min(tr), max(tr), 300)
  spl = make_interp_spline(tr, yr, k=3)
  ys = spl(ts)

  xlabel('Time to maturity'), ylabel('Yield to maturity'), grid(True)
  plt.plot(ts, ys, label=today.strftime("%Y-%m-%d"))
  legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.scatter(tr, yr, marker='x')
plt.title('5-year Yield Curve')
plt.show()


# #### Construct spot rate treasury curve
# 
# The bootstrapping method is built on the following:
# 
# 
# *   For $0 < t_{1} < \frac{1}{2}$
# $$P(0, t_{1}) = (Coupon + Par) \cdot e^{-r(t_{1}) \cdot t_{1}}$$
# *   For $0 < t_{1} < \frac{1}{2} < t_{2} < 1$
# $$P(0, t_{2}) = Coupon \cdot e^{-r(t_{1}) \cdot t_{1}} + (Coupon + Par) \cdot e^{-r(t_{2}) \cdot t_{2}}$$
# *   For $0 < t_{1} < \frac{1}{2} < t_{2} < 1 < t_{3} < \frac{3}{2}$
# $$P(0, t_{2}) = Coupon \cdot e^{-r(t_{1}) \cdot t_{1}} + Coupon \cdot e^{-r(t_{2}) \cdot t_{2}} + (Coupon + Par) \cdot e^{-r(t_{3}) \cdot t_{3}}$$
# *   $\cdots$
# *   For all $t_{i} \in \{TTM, TTM-0.5, TTM-1, ..., TTM \bmod 0.5\}$ 
# 
# Hence the pseudo code is
# 
#     Create an empty list storing time to maturity;
#     Create an empty list storing calculated spot rates;
#     for each bond:
#       Get its timeToMaturity from the current date;
#       Calculate its semi-annual coupon;
#       Calculate its dirty price;
#       
#       # Bootstrapping Spot Curve
#       couponSum = 0;
#       if ttm > (ttm % 0.5): # non-zero-coupon bonds
#         for j in TTM-0.5, TTM-1, ..., TTM mod 0.5:
#           couponSum += discounted coupons by the spot rate on day j;
#       spot = - log((dirtyPrice - couponSum) / (coupon + par)) / timeToMaturity;
#       
#       Store current timeToMaturity into list;
#       Store current spot rate into list;
#     Plot the spot rate curve with interpolation.
# Repeat the above procedure for data of each day and then superimpose the plots.

# In[39]:


# Superimpose spot curves
for today in close['Date']:
    tr, rr = [], []
    # today = datetime.datetime(2022, 1, 10)
    bonds['ttm'] = [(maturity - today).days / 365 for maturity in bonds['Maturity Date']]
    bonds = bonds.sort_values(by=['ttm'])
    # print(bonds)

    for i, bond in bonds.iterrows():
        par = 100
        ttm = bond['ttm']
        coupon = par * bond['Coupon'] / 2.  # semi-annual coupon payment
        price = (0.5 - ttm % 0.5) * bond['Coupon'] + close.iloc[close.loc[close['Date'] == today].index[0], i + 1]

        # Bootstrap
        couponSum = 0

        if ttm > (ttm % 0.5):  # non-zero-coupon bonds
            for j in np.arange(ttm - 0.5, 0, -0.5):
                couponSum += coupon * np.exp(- rr[int(j // 0.5)] * j)

        spot = - np.log((price - couponSum) / (coupon + par)) / ttm
        # print('Bond ' + str(i) + ': ' + str(spot))
        tr.append(ttm)
        rr.append(spot)

    ts = np.linspace(1, max(tr), 300)
    spl = make_interp_spline(tr, rr, k=3)
    rs = spl(ts)

    xlabel('Time to maturity'), ylabel('Spot Rate'), grid(True)
    plt.plot(ts, rs, label=today.strftime("%m/%d/%Y"))
    legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.scatter(np.delete(tr, [0, 1]), np.delete(rr, [0, 1]), marker='x')
plt.title('5-year Spot Curve')
plt.show()


# #### Construct forward rate treasury curve
# 
# Forward rates are calculated from spot rates.
# 
# $$Forward Rate= \frac{(1+r_{2})^{t_{2}}}{(1+r_{1})^{t_{1}}} − 1$$
# where:<br>
# $r_{2} = $ The spot rate for the bond of term $t_{2}$ periods
# <br>
# $r_{1} = $ The spot rate for the bond of term $t_{1}$ periods with $t_{1} < t_{2}$

# The pseudo code is
# 
#     for data of each day do
#         Repeat algorithm 1 to generate enough points {(t1, r1), (t2, r2), ...} from the spot curve
#         for each point do 
#             derive forward rate by f(0, t1, t2) = (1+r2)^t2/(1+r1)^t1 -1
#         end 
#         Plot the forward rate curve over the required interval
#     end
# 
# Repeat the above procedure for data of each day and then superimpose the plots.

# In[40]:


# Superimpose Forward Curves
for today in close['Date']:
    tr, rr = [], []
    # today = datetime.datetime(2022, 1, 10)
    bonds['ttm'] = [(maturity - today).days / 365 for maturity in bonds['Maturity Date']]
    bonds = bonds.sort_values(by=['ttm'])
    # print(bonds)

    for i, bond in bonds.iterrows():
        par = 100
        ttm = bond['ttm']
        coupon = par * bond['Coupon'] / 2.  # semi-annual coupon payment
        price = (0.5 - ttm % 0.5) * bond['Coupon'] + close.iloc[close.loc[close['Date'] == today].index[0], i + 1]

        # Bootstrap
        couponSum = 0

        if ttm > (ttm % 0.5):  # non-zero-coupon bonds
            for j in np.arange(ttm - 0.5, 0, -0.5):
                couponSum += coupon * np.exp(- rr[int(j // 0.5)] * j)

        spot = - np.log((price - couponSum) / (coupon + par)) / ttm
        # print('Bond ' + str(i) + ': ' + str(spot))
        tr.append(ttm)
        rr.append(spot)

    ts = np.linspace(1, max(tr), 300)
    spl = make_interp_spline(tr, rr, k=3)
    rs = spl(ts)

    tfs, fs = [], []
    r1 = rs[0]  # Construct 1-year forward curve
    t1 = 1
    for i in range(np.where(ts > 1.97)[0][0], 300):  # Term ranges from 2 to 5 years
        temp = (1 + rs[i]) ** ts[i] / (1 + r1) ** t1 - 1
        tfs.append(ts[i])
        fs.append(temp)

    xlabel('Time to maturity'), ylabel('Forward Rate'), grid(True)
    plt.plot(tfs, fs, label=today.strftime("%m/%d/%Y"))
    legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('1-year Forward Curve')
plt.show()


# #### Time series and covariance matrices

# In[41]:


# First, we obtain the time series of i-year yield for i = 1, . . . , 5.
list_2d = []

for today in close['Date']:
  tr, yr = [], []
  # today = datetime.datetime(2022, 1, 10)
  bonds['ttm'] = [(maturity - today).days / 365 for maturity in bonds['Maturity Date']]
  bonds = bonds.sort_values(by=['ttm'])
  # print(bonds)

  for i, bond in bonds.iterrows():
      par = 100
      ttm = bond['ttm']
      coupon = par * bond['Coupon'] / 2 # semi-annual coupon payment
      price = (0.5 - ttm % 0.5) * bond['Coupon'] + close.iloc[close.loc[close['Date'] == today].index[0], i+1]
      def RHS_func(ytm): # ttm, coupon, par, PRICE
        if ttm <= (ttm % 0.5): # zero-coupon bond maturing soon
          return (coupon + par) * np.exp(-ttm * ytm) - price
        else: # non-zero-coupon bonds
          temp = 0
          for j in np.arange(ttm, 0, -0.5):
            temp += coupon * np.exp(-j * ytm)
          return temp + par * np.exp(-ttm * ytm) - price
      ytm = optimize.newton(RHS_func, 0.01)
      # print('Bond ' + str(i) + ': ' + str(ytm))
      tr.append(ttm)
      yr.append(ytm)


  ts = np.linspace(1, max(tr), 300)
  spl = make_interp_spline(tr, yr, k=3)
  ys = spl(ts)

  ya = []
  for i in [np.where(ts > 0.97)[0][0], np.where(ts > 1.97)[0][0], np.where(ts > 2.97)[0][0], np.where(ts > 3.97)[0][0], np.where(ts > 4.97)[0][0]]:
    ya.append(ys[i])
  list_2d.append(ya)

column_names = [str(i) + '-year yield' for i in [1, 2, 3, 4, 5]]
df_yield_raw = pd.DataFrame(list_2d, columns = column_names)
print(df_yield_raw)


# In[42]:


# Then we calculate the daily log-returns & the required covariance matrix
# Note that our original data has dates in reverse chronological order
df_yield = pd.DataFrame()
for i in [1, 2, 3, 4, 5]:
    df_yield[str(i) + '-year yield log'] = np.log(df_yield_raw[str(i) + '-year yield'].shift(1)) - np.log(df_yield_raw[str(i) + '-year yield'])

df_yield = df_yield.dropna(axis = 0)
print(df_yield)
print(df_yield.cov())


# In[43]:


# Similarly for forward rates 1yr-1yr, 1yr-2yr, 1yr-3yr, 1yr-4yr.
list_2d = []

for today in close['Date']:
  tr, rr = [], []
  # today = datetime.datetime(2022, 1, 10)
  bonds['ttm'] = [(maturity - today).days / 365 for maturity in bonds['Maturity Date']]
  bonds = bonds.sort_values(by=['ttm'])
  # print(bonds)


  for i, bond in bonds.iterrows():
      par = 100
      ttm = bond['ttm']
      coupon = par * bond['Coupon'] / 2. # semi-annual coupon payment
      price = (0.5 - ttm % 0.5) * bond['Coupon'] + close.iloc[close.loc[close['Date'] == today].index[0], i+1]
      
      # Bootstrap
      couponSum = 0

      if ttm > (ttm % 0.5): #non-zero-coupon bonds
        for j in np.arange(ttm - 0.5, 0, -0.5):
          couponSum += coupon * np.exp(- rr[int(j//0.5)] * j)
    
      spot = - np.log((price - couponSum) / (coupon + par)) / ttm
      #print('Bond ' + str(i) + ': ' + str(spot))
      tr.append(ttm)
      rr.append(spot)


  ts = np.linspace(1, max(tr), 300)
  spl = make_interp_spline(tr, rr, k=3)
  rs = spl(ts)

  tfs, fs = [], []
  r1 = rs[0] # Construct 1-year forward curve
  t1 = 1
  for i in range(np.where(ts > 1.97)[0][0], 300): # Term ranges from 2 to 5 years
    temp = (1+rs[i])**ts[i] / (1+r1)**t1 - 1
    tfs.append(ts[i])
    fs.append(temp)

  fa = []
  for i in [np.where(np.asarray(tfs) > 1.97)[0][0], np.where(np.asarray(tfs) > 2.97)[0][0], np.where(np.asarray(tfs) > 3.97)[0][0], np.where(np.asarray(tfs) > 4.97)[0][0]]:
    fa.append(fs[i])
  list_2d.append(fa)

column_names = ['1yr-' + str(i) + '-yr' for i in [1, 2, 3, 4]]
df_forward_raw = pd.DataFrame(list_2d, columns = column_names)
print(df_forward_raw)


# In[44]:


df_forward = pd.DataFrame()
for i in [1, 2, 3, 4]:
    df_forward['1yr-' + str(i) + '-yr log'] = np.log(df_forward_raw['1yr-' + str(i) + '-yr'].shift(1)) - np.log(df_forward_raw['1yr-' + str(i) + '-yr'])

df_forward = df_forward.dropna(axis = 0)
print(df_forward)
print(df_forward.cov())


# #### PCA

# In[45]:


w, v = LA.eig(df_yield.cov())
print(w, v)
print('The first principal component of the yields is: ', v[np.argmax(w)])


# In[46]:


w, v = LA.eig(df_forward.cov())
print(w, v)
print('The first principal component of the forward rates is: ', v[np.argmax(w)])


# In[ ]:




