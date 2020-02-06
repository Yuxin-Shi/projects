import numpy as np
import scipy.optimize as optimize
from datetime import datetime


class Bond(object):
    def __init__(self, _coupon, _maturity_date, _close_price):
        self.coupon = float(_coupon.split('%')[0])
        self.maturity_date = _maturity_date
        self.close_price = list(reversed(_close_price))
        self.ttm = self.time_to_maturity()
        self.ytm = self.ytm_list()
        self.ytm_average = sum(self.ytm) / 10

    def time_to_maturity(self):
        y = datetime.strptime('01/02/2020', '%m/%d/%Y')
        z = datetime.strptime(self.maturity_date, '%m/%d/%Y')
        diff = z - y
        return diff.days

    def yield_to_maturity(self, n, _close_price):
        ttm = int(self.ttm - n)
        y = int(ttm / 182)
        init = (ttm % 182) / 365
        time = np.asarray([2 * init + n for n in range(0, y + 1)])
        coupon = self.coupon / 2
        accrued_interest = coupon * ((182 - ttm % 182) / 365)
        dirty_price = float(_close_price) + accrued_interest
        pmt = np.asarray([coupon] * y + [coupon + 100])
        ytm_func = lambda y: np.dot(pmt, (1 + y / 2) ** (-time)) - dirty_price
        ytm = optimize.fsolve(ytm_func, 0.05)
        return ytm

    def ytm_list(self):
        ret = []
        for i in range(10):
            ytm = self.yield_to_maturity(i, self.close_price[i])
            ret.append(ytm)
        return ret

