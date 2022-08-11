from nelson_siegel_svensson import NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols

beta0=3.6098917559877353
beta1=-1.838008628009077
beta2=-7.144530936762603e-06
tau=3.7965652566968324


y = NelsonSiegelCurve(beta0, beta1, beta2, tau)

def yield_spread(ytm, CGB_yield):
  """
  CBG_yield = y(Maturity)
  """
  return ytm - CGB_yield/100

"""
from jupyter_files import Coupon_bond,y,yield_spread,


#计算ytm, 对应网站上的yield

a = Coupon_bond()
ytm = a.get_ytm(101.3925, 100, 6.3, 1.61095890410959)

#计算对应国债yield
CGB_yield = y(1.61095890410959)

#计算yield spread， 对应网站的yield spread
y_spread = yield_spread(ytm,CGB_yield/100)

"""
