import numpy as np
import streamlit as st
from scipy.stats import norm, binom, ttest_rel, t, truncnorm, nct, multivariate_t, beta, nct, cauchy, ttest_rel
from scipy.integrate import quad

def logjoint(var, par, mu = 0.5, sigma = 1):
    n1, k1, n2, k2, H0 = par
    beta, psi = var

    p1 = np.exp(beta-(psi/2))/(1+np.exp(beta-(psi/2)))
    p2 = np.exp(beta+(psi/2))/(1+np.exp(beta+(psi/2)))
        
    logbin1 = binom.logpmf(k1, n1, p1)
    logbin2 = binom.logpmf(k2, n2, p2)
        
    if H0:
        log_joint = logbin1 +logbin2 + norm.logpdf(beta) 
    else:            
        log_joint = logbin1 +logbin2 + norm.logpdf(beta) + norm.logpdf(psi, mu, sigma)

    return log_joint

def der_x(x, fun, par, eps = 1e-4):
    f1 = fun([x[0]+eps, x[1]], par)
    f2 = fun([x[0]-eps, x[1]], par)
    return (f1-f2)/(2*eps)

def der_y(x, fun, par, eps = 1e-4):
    f1 = fun([x[0], x[1]+eps], par)
    f2 = fun([x[0], x[1]-eps], par)
    return (f1-f2)/(2*eps)

def der_xy(x, fun, par, eps = 1e-4):
    f1 = der_x([x[0], x[1]+eps], fun, par)
    f2 = der_x([x[0], x[1]-eps], fun, par)
    return (f1-f2)/(2*eps)

def der_xx(x, fun, par, eps = 1e-4):
    f1 = der_x([x[0]+eps, x[1]], fun, par)
    f2 = der_x([x[0]-eps, x[1]], fun, par)
    return (f1-f2)/(2*eps)

def der_yy(x, fun, par, eps = 1e-4):
    f1 = der_y([x[0], x[1]+eps], fun, par)
    f2 = der_y([x[0], x[1]-eps], fun, par)
    return (f1-f2)/(2*eps)

def Hessian(x, fun, par, eps = 1e-4):
    return np.array([[der_xx(x, fun, par), der_xy(x, fun, par)], 
                    [der_xy(x, fun, par), der_yy(x, fun, par)]])

def sigma_sq(x, fun, par, eps = 1e-4):
    second_derivative = der_xx(x, fun, par)
    return -1/second_derivative

def mode_logjoint(par, H0=True):
    beta = 0
    psi = 0
    beta_old = -100
    psi_old = -100
    if H0:
        while np.abs(beta_old - beta) > 0.01:
            beta_old = beta
            db = der_x([beta, psi], logjoint, par)
            d2b = der_xx([beta, psi], logjoint, par)
            beta = beta - (db/d2b)
            log_joint = logjoint([beta, psi], par)
            
    else:
        while np.abs(beta_old - beta) > 0.01 or np.abs(psi_old - psi) > 0.01:
            beta_old = beta
            psi_old = psi
            db = der_x([beta, psi], logjoint, par)
            dp = der_y([beta, psi], logjoint, par)
            d2b = der_xx([beta, psi], logjoint, par)
            d2p = der_yy([beta, psi], logjoint, par)
            beta = beta - (db/d2b)
            psi = psi - (dp/d2p)
            log_joint = logjoint([beta, psi], par)
            
    return beta, psi, log_joint

def lik_H0(data, mu_beta = 0, sigma_beta = 1):
    par = data + [True]
    beta_star, _, l_star = mode_logjoint(par)
    sigma0 = sigma_sq([beta_star, 0], logjoint, par)
    return np.sqrt(2*np.pi*sigma0)*np.exp(l_star)

def lik_H1(data, mu_beta = 0, sigma_beta = 1, mu_psi = 0, sigma_psi = 1):
    par = data + [False]
    beta_star, psi_star, l_star = mode_logjoint(par, H0 = False)
    
    H1 = Hessian([beta_star, psi_star], logjoint, par)
    sigma1 = np.linalg.inv(-H1)
    
    lik = 2*np.pi*np.sqrt(np.linalg.det(sigma1))*np.exp(l_star)
    
    return lik, psi_star, sigma1[1, 1]

def posterior(data):
    L0 = lik_H0(data)
    L1, psi, sigma = lik_H1(data)

    Bm0 = (t.cdf(0, loc = psi, scale = sigma, df=5)/0.5)*(L1/L0)
    Bp0 = ((1-t.cdf(0, loc = psi, scale = sigma, df=5))/0.5)*(L1/L0)
    #print(Bm0, Bp0)

    post_0P = Bp0/(Bp0+1)
    return Bm0, Bp0, post_0P

def main_AB(N, a = 18, b = 2, n_raters = 9, n_sim = 1000, threshold_post = 0.95, threshold_BF = np.sqrt(10)):
    post = []
    BF = []
    for i in range(n_sim):
        #p = beta.rvs(a, b, size = N)
        p = beta.rvs(a, b)
        attempt1 = N
        attempt2 = n_raters*N     
        success1 = np.sum(np.random.choice([0,1], size = N, p = [1-p, p]))
        success2 = np.sum(np.random.choice([0,1], size = n_raters*N, p = [1-p, p]))
        
        data = [attempt1, success1, attempt2, success2]
        Bm0, Bp0, post_plus = posterior(data)
        post_0P = 1 - post_plus
        BF.append(1/Bp0)
        post.append(post_0P)

    return np.mean(post), np.mean(np.array(post)>=threshold_post), np.mean(BF), np.mean(np.array(BF)>=threshold_BF)

def numerator(delta, tstat, n):
    df = np.min([n-1, 100])
    return (nct.pdf(tstat, df, np.sqrt(n)*delta))*(cauchy.pdf(delta, 0, 0.707))

def lognum(delta, tstat, n):
    df = np.min([n-1, 100])
    return nct.logpdf(tstat, df, np.sqrt(n)*delta) + cauchy.logpdf(delta, 0, 0.707)
    
def find_mode(tstat, n, low, high, step=0.01):
    # Generate random input values within the specified range
    x_samples = np.arange(low, high+step, step)
    
    # Evaluate the function at each of the sampled input values
    y_samples = np.array([numerator(x, tstat, n) for x in x_samples])
    
    mask = np.isnan(y_samples)
    y_samples = y_samples[~mask]
    x_samples = x_samples[~mask] 
    
    mask = np.isinf(np.abs(y_samples))
    y_samples = y_samples[~mask]
    x_samples = x_samples[~mask] 

    # Find the input value that corresponds to the maximum function value
    mode_index = np.argmax(y_samples)
    mode = x_samples[mode_index]
    
    return mode


def integral_limits(data, tstat, start=1e-2, step=1e-2, tolerance=1e-15):
    delta = np.mean(data)
    n = len(data)  
    mode = find_mode(tstat, n, -np.abs(delta)-1, np.abs(delta)+1)  
    #print(mode)
    # Optimize lower bound
    diff = 100
    prev_integral = 0
    lower_limit = mode - start
    while True:
        integral_value, error = quad(numerator, lower_limit, mode, args=(tstat, n))
        diff = np.abs(integral_value - prev_integral)
        if diff < tolerance:
            break
        prev_integral = integral_value
        lower_limit -= step
       
    # Optimize upper bound
    upper_limit = mode + (mode-lower_limit)
    #prev_integral = 0
    #diff = 100
    #upper_limit = mode + start
    #while True:
    #    integral_value, error = quad(numerator, mode, upper_limit, args=(tstat, n))
    #    diff = np.abs(integral_value - prev_integral)
    #    if diff < tolerance:
    #        break
    #    prev_integral = integral_value
    #    upper_limit += step
    integral_value, error = quad(numerator, lower_limit, upper_limit, args=(tstat, n))
    return integral_value, lower_limit+step, upper_limit-step
        
    
def Bayes_factor_10(data1, data2):
    data = (data1-data2)/np.std(data1-data2)
    delta = np.mean(data)
    n = len(data)
    tstat = ttest_rel(data1, data2).statistic 
    num, low, high = integral_limits(data, tstat)
    #print(low, high)
    den = t.pdf(tstat, n-1)
    return num/den, low, high

def Bayes_factor_plus0(data1, data2):
    data = (data1-data2)/np.std(data1-data2)
    delta = np.mean(data)
    n = len(data)
    tstat = ttest_rel(data1, data2).statistic
    B10, low, high = Bayes_factor_10(data1, data2)
    if low > 0:
        return np.inf
    elif high < 0:
        return 0
    else:
        pos = quad(numerator, 0, high, args=(tstat, n))[0]
        tot = quad(numerator, low, high, args=(tstat, n))[0]
        Bplus1 = (pos/tot)/0.5
        return B10*Bplus1
    
def main_t(N, n_sim = 1000, threshold_post = 0.95, threshold_BF = np.sqrt(10)):
    post = []
    BF = []
    for _ in range(n_sim):
        data1 = norm.rvs(size = N)
        data2 = norm.rvs(size = N)
        Bplus0 = Bayes_factor_plus0(data1, data2)
        if Bplus0 == np.inf:
            post.append(0)
            BF.append(0)
        elif Bplus0 == 0:
            post.append(1)
            BF.append(np.inf)
        else:
            post.append(1-(Bplus0/(1+Bplus0)))
            BF.append(1/Bplus0)
    return np.mean(post), np.mean(np.array(post)>=threshold_post), np.mean(BF), np.mean(np.array(BF)>=threshold_BF)

##### Main function #######

def main():
    st.set_page_config(
        page_title="Bayesian sample calculator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    N_t = st.number_input("Sample size for t-test", value=100)
    n_sim_t = st.number_input("Number of simulations  for t-test", value=1000)
    threshold_post_t = st.number_input("Threshold for posterior probability  for t-test", value = 0.95)
    threshold_BF_t = st.number_input("Threshold for Bayes Factor  for t-test", value = np.sqrt(10))

    if st.button('Calculate ratios for t-test'):
        post_average, post_threshold, BF_average, BF_threshold = main_t(N_t, n_sim_t, threshold_post_t, threshold_BF_t)
        st.write(f"With a sample size of {N_t}, you can expect an average posterior of {post_average}")
        st.write(f"With a sample size of {N_t}, you can expect your posterior to be over {threshold_post_t} with a confidence of {post_threshold}")
        st.write(f"With a sample size of {N_t}, you can expect an average Bayes Factor of {BF_average}")
        st.write(f"With a sample size of {N_t}, you can expect yout Bayes Factor to be over {threshold_BF_t} with a confidence of {BF_threshold}")


    N_AB = st.number_input("Sample size for AB test", value=100)
    n_sim_AB = st.number_input("Number of simulations for AB test", value=1000)
    threshold_post_AB = st.number_input("Threshold for posterior probability  for AB test", value = 0.95)
    threshold_BF_AB = st.number_input("Threshold for Bayes Factor  for AB test", value = np.sqrt(10))
    a = st.number_input("Preliminary successes", value = 18)
    b = st.number_input("Preliminary failures", value = 2)
    n_raters = st.number_input("Number of raters", value = 9)

    if st.button('Calculate ratios for AB test'):
        post_average, post_threshold, BF_average, BF_threshold = main_AB(N_AB, a, b, n_raters, n_sim_AB, threshold_post_AB, threshold_BF_AB)
        st.write(f"With a sample size of {N_AB}, you can expect an average posterior of {post_average}")
        st.write(f"With a sample size of {N_AB}, you can expect your posterior to be over {threshold_post_AB} with a confidence of {post_threshold}")
        st.write(f"With a sample size of {N_AB}, you can expect an average Bayes Factor of {BF_average}")
        st.write(f"With a sample size of {N_AB}, you can expect yout Bayes Factor to be over {threshold_BF_AB} with a confidence of {BF_threshold}")


if __name__ == '__main__':
    main()
