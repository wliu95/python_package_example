from nelson_siegel_svensson import NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols

beta0=3.6098917559877353
beta1=-1.838008628009077
beta2=-7.144530936762603e-06
tau=3.7965652566968324


y = NelsonSiegelCurve(beta0, beta1, beta2, tau)
