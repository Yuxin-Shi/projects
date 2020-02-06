import os
import pandas as pd
import matplotlib.pyplot as plt
from Bond import *


folder_path = os.getcwd()
file_path = os.path.join(folder_path, '12bonds.csv')
data = pd.read_csv(file_path)


def ttm(x):
    current_date = x['Date'][0]
    x['time to maturity'] = [(datetime.strptime(maturity, '%m/%d/%Y') - datetime.strptime(current_date, '%m/%d/%Y')).days for maturity in x['Maturity_date']]


def ytm(x):
    tr, yr = [], []
    for i, bond in x.iterrows():
        ttm = bond['time to maturity']
        tr.append(ttm / 365)
        y = int(ttm / 182)
        init = (ttm % 182) / 365
        time = np.asarray([2 * init + n for n in range(0, y + 1)])
        coupon = float(bond['Coupon'].split('%')[0]) * 100 / 2
        accrued_interest = coupon * ((182 - ttm % 182) / 365)
        dirty_price = bond['Close_price'] + accrued_interest
        pmt = np.asarray([coupon] * y + [coupon + 100])
        ytm_func = lambda y: np.dot(pmt, (1 + y / 2) ** (-time)) - dirty_price
        ytm = optimize.fsolve(ytm_func, .05)
        yr.append(ytm)
    return tr, yr


def spot(x):
    s = np.empty([1, 10])
    tr = []
    coupons = []
    dirty_price = []
    for i, bond in x.iterrows():
        ttm = bond['time to maturity']
        tr.append(ttm / 365)
        coupon = float(bond['Coupon'].split('%')[0]) * 100 / 2
        coupons.append(coupon)
        accrued_interest = coupon * (0.5 - (ttm % 182) / 365)
        dirty_price.append(bond['Close_price'] + accrued_interest)

    for i in range(0, 10):
        if i == 0:

            s[0, i] = -np.log(dirty_price[i] / (coupons[i] + 100)) / tr[i]
        else:
            pmt = np.asarray([coupons[i]] * i + [coupons[i] + 100])
            spot_func = lambda y: np.dot(pmt[:-1],                                         np.exp(-(np.multiply(s[0, :i], tr[:i])))) + pmt[i] * np.exp(-y * tr[i]) - dirty_price[i]
            s[0, i] = optimize.fsolve(spot_func, 0.1)
    return tr, s


def forward(x):
    s = spot(x)[1].squeeze()
    f1 = (s[3] * 2 - s[1] * 1) / (2 - 1)
    f2 = (s[5] * 3 - s[1] * 1) / (3 - 1)
    f3 = (s[7] * 4 - s[1] * 1) / (4 - 1)
    f4 = (s[9] * 5 - s[1] * 1) / (5 - 1)
    f = [f1, f2, f3, f4]
    return f

days = ['2020/01/02', '2020/01/03', '2020/01/06', '2020/01/07', '2020/01/08',
        '2020/01/09', '2020/01/10', '2020/01/13', '2020/01/14', '2020/01/15']


Bond_list = []
close_price = []
df = []
for index, row in data.iterrows():
    close_price.append(row['Close_price'])
    if index % 10 == 9:
        coupon = row['Coupon']
        maturity_date = row['Maturity_date']
        Bond_list.append(Bond(coupon, maturity_date, close_price))
        close_price = []

for i in range(10):
    p = [10 * j + i for j in range(10)]
    tmp = data.iloc[p]
    tmp = tmp.reset_index(drop=True)
    print(tmp)
    df.append(tmp)

for i in range(10):
    print(Bond_list[i].ytm_average[0])


plt.figure()
plt.xlabel('time to maturity')
plt.ylabel('Yield')
plt.title('5-year yield curve')
for i in range(1, 10):
    x = [(Bond_list[j].ttm - i) / 365 for j in range(10)]
    y = [Bond_list[j].ytm[i]+ 0.0005 * i for j in range(10)]
    print(y)
    plt.plot(x, y, label=days[i])
plt.legend()
# Answer plot to Q4(a)
plt.show()

log_X = np.empty([5, 9])
y = np.empty([5, 10])
for i in range(len(df)):
    ttm(df[i])
    y[0, i] = ytm(df[i])[1][1]
    y[1, i] = ytm(df[i])[1][3]
    y[2, i] = ytm(df[i])[1][5]
    y[3, i] = ytm(df[i])[1][7]
    y[4, i] = ytm(df[i])[1][9]

for i in range(0, 9):
    log_X[0, i] = np.log(y[0, i + 1] / y[0, i])
    log_X[1, i] = np.log(y[1, i + 1] / y[1, i])
    log_X[2, i] = np.log(y[2, i + 1] / y[2, i])
    log_X[3, i] = np.log(y[3, i + 1] / y[3, i])
    log_X[4, i] = np.log(y[4, i + 1] / y[4, i])


cov1 = np.cov(log_X)
# Answer to Q5
print(cov1)

f = np.empty([4, 10])
for i in range(len(df)):
    f[0, i] = forward(df[i])[0]
    f[1, i] = forward(df[i])[1]
    f[2, i] = forward(df[i])[2]
    f[3, i] = forward(df[i])[3]

cov2 = np.cov(f)
# Answer to Q5
print(cov2)

w, v = np.linalg.eig(np.cov(log_X))
# Answer to Q6
print(w,v)

a, b = np.linalg.eig(np.cov(f))
# Answer to Q6
print(a,b)