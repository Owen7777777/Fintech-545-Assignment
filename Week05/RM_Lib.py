# Import necessary library

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, t, norm, multivariate_normal, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy.random as npr
from scipy.stats import norm
from scipy.integrate import quad

# Covariance Estimation
def cov_skip_miss(df):
    df_skip = df.dropna()
    return np.cov(df_skip, rowvar = False)

def corr_skip_miss(df):
    df_skip = df.dropna()
    return df_skip.corr().values

def cov_pairwise(df):
    cov_matrix = df.cov(min_periods=1)
    return cov_matrix

def corr_pairwise(df):
    corr_matrix = df.corr(min_periods=1)
    return corr_matrix.values

def ewCovar(x, lbda):
    if type(x) != np.ndarray:
        x = x.values
    m, n = x.shape
    w = np.empty(m)
    
    # Remove the mean from the series
    xm = np.mean(x, axis=0)
    x = (x - xm)
    
    # Calculate weight. Realize we are going from oldest to newest
    w = (1 - lbda) * lbda ** np.arange(m)[::-1]
    
    # Normalize weights to 1
    w /= np.sum(w)
    
    w = w.reshape(-1, 1)
    
    # covariance[i,j] = (w * x.T) @ x
    return (w * x).T @ x

def ewCorr(x, lbda):
    cov = ewCovar(x, lbda)
    invSD = np.diag(1.0 / np.sqrt(np.diag(cov)))
    corr = np.dot(invSD, cov).dot(invSD)
    return corr

# Covarariance matrix based on different ew variance and ew correlation
def cov_with_different_ew_var_corr(df, ew_var_lbda, ew_corr_lbda):
    ew_cov = ewCovar(df, ew_var_lbda)
    ew_var = np.diag(np.diag(ew_cov))
    invSD =  np.sqrt(ew_var) # ew_std
    
    ew_corr = ewCorr(df, ew_corr_lbda)
    cov = np.dot(invSD, ew_corr).dot(invSD)
    return cov


#Non-PSD fixes for correlation matrices
# Rebonato and Jackel method

# near_psd
def RJ_nearestPSD(a, epsilon=0.0):
    
    # 考虑了输入是cov或者corr的两种情况
    n = a.shape[0]
    invSD = None
    
    # 如果 a 不是相关矩阵，则转换为相关矩阵
    if not np.allclose(np.diag(a), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(a)))
        a = np.dot(invSD, a).dot(invSD)
    
    # 计算特征值和特征向量
    vals, vecs = np.linalg.eigh(a)
    
    # 修正特征值
    vals = np.maximum(vals, epsilon)
    
    # 计算 T
    T = 1.0 / np.dot(vecs**2,vals)

    T = np.diag(T)

    # 计算 l
    l = np.diag(np.sqrt(vals))
    
    # 计算 B
    B = np.sqrt(T).dot(vecs).dot(l)
    
    # 计算近似 PSD 矩阵
    a_psd = B.dot(B.T)
    
    # 如果之前转换了相关矩阵，则反转这个转换
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        a_psd = invSD.dot(a_psd).dot(invSD)

    return a_psd

# higham near_psd

# Helper Function
def _getAplus(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.diag(np.maximum(vals, 0))
    return np.dot(vecs, np.dot(vals, vecs.T))

def _getPS(A, W):
    W05 = np.sqrt(W)
    iW05 = np.linalg.inv(W05)
    return np.dot(iW05, np.dot(_getAplus(np.dot(W05, np.dot(A, W05))), iW05))

def _getPu(A, W):
    Aret = A.copy()
    np.fill_diagonal(Aret, 1)
    return Aret

def wgtNorm(A, W):
    W05 = np.sqrt(W)
    WA = W05.dot(A).dot(W05)
    W_norm = np.sum(WA**2)
    return np.sum(W_norm)

# pc means pearson correlation
def higham_nearestPSD(pc, W=None, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    
    # 如果 pc 不是相关矩阵，则转换为相关矩阵
    invSD = None
    if not np.allclose(np.diag(pc), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(pc)))
        pc = np.dot(invSD, pc).dot(invSD)
        
    if W is None:
        W = np.diag(np.ones(n))
    
    Yk = pc.copy()
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS if i > 1 else Yk
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W)
        norm = wgtNorm(Yk - pc, W)
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if abs(norm - norml) < tol and minEigVal > -epsilon:
            break

        norml = norm
        i += 1

    if i < maxIter:
        print("Converged in {} iterations.".format(i))
    else:
        print("Convergence failed after {} iterations".format(i - 1))
            
    # 如果之前转换了相关矩阵，则反转这个转换
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        Yk = invSD.dot(Yk).dot(invSD)

    return Yk

# Simulation Methods
def chol_psd(a):
    if type(a) != np.ndarray:
        a = a.values
        
    n = a.shape[0]  # 获取矩阵a的行数
    root = np.zeros([n, n])  # 将root矩阵初始化为全0矩阵

    for j in range(n):  # 对于矩阵a的每一列进行循环
        s = 0.0
        if j > 0:  # 如果不是第一列，需要计算之前列的点积
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:  # 如果temp的值接近0，则设置为0
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:  # 如果对角元素为0，即当前列全为0
            root[(j+1):n,j] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j+1, n):
                s = np.dot(root[i, :j], root[j, :j].T)
                root[i, j] = (a[i, j] - s) * ir
                
    return root

def simulate_normal(N, cov, mean=None, seed=1234):
    n = cov.shape[0]
    if cov.shape[1] != n:
        raise ValueError(f"Covariance matrix is not square ({n},{cov.shape[1]})")

    if mean is None:
        mean = np.zeros(n)
    elif mean.shape[0] != n:
        raise ValueError(f"Mean ({mean.shape[0]}) is not the size of cov ({n},{n})")

    # Take the root of the covariance matrix
#     l = np.zeros([n,n])
#     chol_psd(l, cov)  
    l = chol_psd(cov) 

    # Generate needed random standard normals
    npr.seed(seed)
    out = npr.standard_normal((N, n))

    # Apply the Cholesky root to the standard normals
    out = np.dot(out, l.T)

    # Add the mean
    out += mean

    return out

def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    if mean is None:
        mean = np.zeros(n)
    elif mean.shape[0] != n:
        raise ValueError(f"Mean size {mean.shape[0]} does not match covariance size {n}.")

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    # Sort eigenvalues and eigenvectors
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Calculate total variance
    tv = np.sum(vals)

    # Select principal components based on pctExp
    cum_var_exp = np.cumsum(vals) / tv
    if pctExp < 1:
        n_components = np.searchsorted(cum_var_exp, pctExp) + 1 # 这个函数在cum_var_exp数组中查找pctExp应该插入的位置
        vals = vals[:n_components]
        vecs = vecs[:, :n_components]
    else:
        n_components = n
    # Construct principal component matrix
    B = vecs @ np.diag(np.sqrt(vals))

    # Generate random samples
    np.random.seed(seed)
    r = np.random.randn(n_components, nsim)
    out = (B @ r).T

    # Add the mean
    out += mean

    return out


# Fitted Model Funtion
class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval = eval_func
        self.errors = errors
        self.u = u

def fit_normal(x):
    
    # 计算均值和标准差
    m = np.mean(x)
    s = np.std(x, ddof = 1)
    
    # 创建正态分布模型
    error_model = norm(m, s)
    
    # 计算误差和累积分布函数值
    errors = x - m
    u = error_model.cdf(x)
    
    # 定义分位数函数
    def eval_u(u):
        return error_model.ppf(u)
    
    # 返回拟合的模型对象
    return FittedModel(None, error_model, eval_u, errors, u)

def fit_general_t(x):
    params = t.fit(x)
    df, loc, scale = params
    error_model = t(df=df, loc=loc, scale=scale)
    
    errors = x - loc
    
    u = error_model.cdf(x)
    
    def eval_u(u):
        return error_model.ppf(u)
    
    fit_model = FittedModel(None, error_model, eval_u, errors, u)
    opt_para = loc, scale, df
    # 返回拟合的模型对象
    return np.array(opt_para), fit_model

def general_t_ll(mu, s, nu, x):
    """计算广义t分布的对数似然和。"""
    td = stats.t(df=nu, loc=mu, scale=s)
    return np.sum(np.log(td.pdf(x)))

def fit_regression_t(y, x):
    if len(x.shape) == 1:
        x = x.values.reshape(-1,1)
    if type(y) != np.ndarray:
        y = y.values
    n = x.shape[0]
    X = np.hstack((np.ones((n, 1)), x))
    nB = X.shape[1]

    # 使用OLS结果作为起始估计值
    b_start = np.linalg.inv(X.T @ X) @ X.T @ y
    e = y - X @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / stats.kurtosis(e, fisher=False) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    # 优化目标函数
    def objective(params):
        m, s, nu, *B = params
        xm = y - X @ np.array(B)
        return -general_t_ll(m, s, nu, xm)

    initial_params = [start_m, start_s, start_nu] + b_start.tolist()
    bounds = [(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * nB
    result = minimize(objective, initial_params, bounds=bounds)

    m, s, nu, *beta = result.x

    # 定义拟合的误差模型
    errorModel = t(df=nu, loc=m, scale=s)

    def eval_model(x, u):
        if len(x.shape) == 1:
            x = x.values.reshape(-1,1)
        n = x.shape[0]
        _temp = np.hstack((np.ones((n, 1)), x))
        return _temp @ np.array(beta) + errorModel.ppf(u)

    # 计算回归误差及其U值
    # 均值部分都要减去？
    errors = y - eval_model(x, np.full(x.shape[0], 0.5))
    u = errorModel.cdf(errors)
    opt_para = result.x
    fit_model = FittedModel(beta, errorModel, eval_model, errors, u)
    return np.array(opt_para), fit_model


# VaR Calculation Methods

def return_calculate(prices, method="DISCRETE", date_column="Date"):
    # Make sure date column exist
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame")

    # Choose all colums excpet for date
    cols = [col for col in prices.columns if col != date_column]
    
    # Extract Price data
    p = prices[cols].values
    n, m = p.shape
    
    # Calculate price ratios at consecutive points in time
    p2 = p[1:, :] / p[:-1, :]
    
    # Calculate rate of return based on method
    if method.upper() == "DISCRETE":
        p2 -= 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")
    
    # Create a DataFrame containing the results
    out = pd.DataFrame(p2, columns=cols)
#     out[date_column] = prices[date_column].values[1:]
    out.insert(loc=0, column=date_column, value=prices[date_column].values[1:])
    
    return out


def VaR_cal(method, ret, PV, Asset_value, holdings, name, current_prices, alpha):

    # Calcualte Covariance Matrix and Portfiolio Volaitility
    if method == "Normal":
        # R_gradients also equal to weights
        R_gradients = np.array(Asset_value) / PV
        Sigma = np.cov(ret, rowvar=False)
        p_sig = np.sqrt(np.dot(R_gradients.T, np.dot(Sigma, R_gradients)))
        VaR = (-PV) * norm.ppf(alpha) * p_sig
    
    elif method == "EW_Normal":
        R_gradients = np.array(Asset_value) / PV
        Sigma = ewCovar(ret,0.94)
        p_sig = np.sqrt(np.dot(R_gradients.T, np.dot(Sigma, R_gradients)))
        VaR = (-PV) * norm.ppf(alpha) * p_sig
    
    elif method == "MLE_T":
        params = stats.t.fit(ret)
        df, loc, scale = params
        VaR = (-PV) * stats.t.ppf(alpha, df, loc, scale)
    
    elif method == "AR_1":
        model = ARIMA(ret, order=(1, 0, 0))
        model_fit = model.fit()
        phi_0 = model_fit.params['const']  # or model_fit.params[0]
        phi_1 = model_fit.params['ar.L1']  # or model_fit.params[1]
        predicted_return = phi_0 + phi_1 * ret.values[-1,0]
        
        # Calculate Std and VaR
        residual_std = model_fit.resid.std()
        VaR = (-PV) * (predicted_return + norm.ppf(alpha) * residual_std)
    
    elif method == "Historical":
        rand_indices = np.random.choice(ret.shape[0], size=10000, replace=True)
        sim_ret = ret.values[rand_indices, :]
        sim_price = current_prices.values * (1 + sim_ret)
        vHoldings = np.array([holdings[nm] for nm in name])
        pVals = sim_price @ vHoldings
        VaR = PV - np.percentile(pVals, alpha * 100)
    return VaR
#     print(f"{method} VaR")
#     print(f"Current Portfolio Value: {PV}")
#     print(f"Current Portfolio VaR: {VaR}")

def simple_VaR(rets, dist, alpha = 0.05, lbda = 0.97):
    if type(rets) != np.ndarray:
        rets = rets.values.reshape(-1,1)
    if dist == "Normal":
        fitted_model = fit_normal(rets) 
        VaR_abs =  -norm.ppf(alpha, fitted_model.error_model.mean(), fitted_model.error_model.std())
        VaR_diff_from_mean = -(-VaR_abs - fitted_model.error_model.mean())
        return np.array([VaR_abs, VaR_diff_from_mean])
    elif dist == "EW_Normal":
        std = np.sqrt(ewCovar(rets,lbda))
        VaR_abs =  -norm.ppf(alpha, np.mean(rets), std)
        VaR_diff_from_mean = -(-VaR_abs - np.mean(rets))
        return np.array([VaR_abs, VaR_diff_from_mean]).reshape(-1)
    elif dist == "T":
        opt_para, fitted_model = fit_general_t(rets)
        VaR_abs = -t.ppf(alpha, df = opt_para[2], loc = opt_para[0], scale = opt_para[1])
        VaR_diff_from_mean = -(-VaR_abs - opt_para[0])
        return np.array([VaR_abs, VaR_diff_from_mean])

def simple_VaR_sim(rets, dist, alpha = 0.05, N = 100000):
    if type(rets) != np.ndarray:
        rets = rets.values
    if dist == "Normal":
        fitted_model = fit_normal(rets)
        rand_num = norm.rvs(fitted_model.error_model.mean(),fitted_model.error_model.std(), size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        VaR_abs = -(xs[iup] + xs[idn]) / 2
        VaR_diff_from_mean = -(-VaR_abs - np.mean(xs))
        return np.array([VaR_abs, VaR_diff_from_mean])
    elif dist == "T":
        opt_para, fit_model = fit_general_t(rets)
        rand_num = t.rvs(df = opt_para[2], loc = opt_para[0], scale = opt_para[1], size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        VaR_abs = -(xs[iup] + xs[idn]) / 2
        VaR_diff_from_mean = -(-VaR_abs - np.mean(xs))
        return np.array([VaR_abs, VaR_diff_from_mean]) 
        
# ES Calculation Methods
def simple_ES(rets, dist, alpha = 0.05, lbda = 0.97):
    if type(rets) != np.ndarray:
        rets = rets.values.reshape(-1,1)
    if dist == "Normal":
        VaR_abs = simple_VaR(rets, dist, alpha)[0]
        fitted_model = fit_normal(rets)
        def integrand(x):
            return x * norm.pdf(x,fitted_model.error_model.mean(), fitted_model.error_model.std())
        integral_abs, error = quad(integrand, -np.inf, -VaR_abs)

        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-fitted_model.error_model.mean())
        return np.array([ES_abs, ES_diff_from_mean])
    
    elif dist == "EW_Normal":
        VaR_abs = simple_VaR(rets, dist, alpha, lbda)[0]
        def integrand(x):
            std = np.sqrt(ewCovar(rets, lbda)[0])
            return x * norm.pdf(x,np.mean(rets), std)
        integral_abs, error = quad(integrand, -np.inf, -VaR_abs)
        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-np.mean(rets))
        return np.array([ES_abs, ES_diff_from_mean])
    
    elif dist == "T":
        VaR_abs = simple_VaR(rets, dist, alpha)[0]
        opt_para, fitted_model = fit_general_t(rets)
        def integrand(x):
            return x * t.pdf(x,df = opt_para[2], loc = opt_para[0], scale = opt_para[1])
        integral_abs, error = quad(integrand, -np.inf, -VaR_abs)

        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-opt_para[0])
        return np.array([ES_abs, ES_diff_from_mean])

def simple_ES_sim(rets, dist, alpha = 0.05, N = 1000000):
    if type(rets) != np.ndarray:
        rets = rets.values
    if dist == "Normal":
        fitted_model = fit_normal(rets)
        rand_num = norm.rvs(fitted_model.error_model.mean(),fitted_model.error_model.std(), size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        ES_abs = -np.mean(xs[0:idn])
        ES_diff_from_mean = -(-ES_abs - np.mean(xs))
        return np.array([ES_abs, ES_diff_from_mean])
    elif dist == "T":
        opt_para, fit_model = fit_general_t(rets)
        rand_num = t.rvs(df = opt_para[2], loc = opt_para[0], scale = opt_para[1], size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        ES_abs = -np.mean(xs[0:idn])
        ES_diff_from_mean = -(-ES_abs - np.mean(xs))
        return np.array([ES_abs, ES_diff_from_mean])


# VaR and ES
def VaR_ES(x, alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (xs[iup] + xs[idn]) / 2
    ES = np.mean(xs[0:idn])
    return -VaR, -ES

def Historical_VaR_ES(rets, size = 10000, alpha = 0.05):
    rand_indices = np.random.choice(rets.shape[0], size, replace=True)
    sim_rets = rets.values[rand_indices, :]
    xs = np.sort(sim_rets, axis = 0)
    n = alpha * len(xs)
    VaR_abs = -np.percentile(sim_rets, alpha * 100)
    VaR_diff_from_mean = - (- VaR_abs - np.mean(sim_rets))
    idn = int(np.floor(n))
    ES_abs = -np.mean(xs[0:idn])
    ES_diff_from_mean = -(-ES_abs - np.mean(sim_rets))
    return np.array([VaR_abs, VaR_diff_from_mean, ES_abs, ES_diff_from_mean])


    

