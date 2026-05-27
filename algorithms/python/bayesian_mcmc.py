# -*- coding: utf-8 -*-
"""
贝叶斯推断 & MCMC — Python 实现
Metropolis-Hastings / Gibbs 采样

用法：
    python bayesian_mcmc.py
"""

import numpy as np
from scipy import stats


def metropolis_hastings(log_posterior, proposal_std, n_samples=10000,
                        burn_in=2000, x0=None, ndim=1):
    """
    Metropolis-Hastings MCMC 采样

    Parameters
    ----------
    log_posterior : callable f(x) -> float
        对数后验分布（unnormalized）
    proposal_std : float or ndarray
        提议分布标准差（高斯随机游走）
    n_samples : int
        采样数
    burn_in : int
        预热期（丢弃）
    x0 : ndarray, optional
        初始值

    Returns
    -------
    dict: samples, acceptance_rate
    """
    if x0 is None:
        x0 = np.zeros(ndim)
    x0 = np.asarray(x0)
    proposal_std = np.asarray(proposal_std)

    samples = np.zeros((n_samples + burn_in, len(x0)))
    samples[0] = x0
    log_post_current = log_posterior(x0)
    n_accept = 0

    for i in range(1, n_samples + burn_in):
        # 提议
        x_proposal = samples[i-1] + np.random.normal(0, proposal_std, len(x0))
        log_post_proposal = log_posterior(x_proposal)

        # 接受概率
        log_alpha = log_post_proposal - log_post_current
        if np.log(np.random.rand()) < log_alpha:
            samples[i] = x_proposal
            log_post_current = log_post_proposal
            n_accept += 1
        else:
            samples[i] = samples[i-1]

    chain = samples[burn_in:]

    return {
        'samples': chain,
        'acceptance_rate': n_accept / (n_samples + burn_in - 1),
        'mean': np.mean(chain, axis=0),
        'std': np.std(chain, axis=0),
        'ci_95': np.percentile(chain, [2.5, 97.5], axis=0),
    }


def bayesian_update(prior_params, likelihood_func, data):
    """
    共轭先验的贝叶斯更新（解析解）

    常见共轭对:
    - Beta-Binomial: Beta(α,β) + Binom → Beta(α+successes, β+failures)
    - Normal-Normal: N(μ₀,τ₀²) + N(μ,σ²/n) → N(...)
    - Gamma-Poisson: Gamma(α,β) + Poisson → Gamma(α+Σx, β+n)

    Returns:
        后验分布参数
    """
    dist_type = prior_params.get('type', '')

    if dist_type == 'beta_binomial':
        alpha_prior, beta_prior = prior_params['alpha'], prior_params['beta']
        successes, n_trials = data['successes'], data['n']
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (n_trials - successes)
        return {'type': 'beta', 'alpha': alpha_post, 'beta': beta_post,
                'posterior_mean': alpha_post / (alpha_post + beta_post)}

    elif dist_type == 'normal_normal':
        mu_0, tau2_0 = prior_params['mu_0'], prior_params['tau2_0']
        sigma2, n = data['sigma2'], data['n']
        if n == 0:
            return prior_params

        precision_prior = 1 / tau2_0
        precision_data = n / sigma2
        precision_post = precision_prior + precision_data
        tau2_post = 1 / precision_post
        mu_post = (precision_prior * mu_0 + precision_data * data['x_bar']) / precision_post

        return {'type': 'normal', 'mu_post': mu_post, 'tau2_post': tau2_post}

    return None


if __name__ == '__main__':
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("贝叶斯 MCMC 示例")
    print("=" * 60)

    # --- 示例1: MCMC 估计正态分布均值 ---
    # 数据生成: N(5, 2²)
    true_mu = 5.0
    data = np.random.normal(true_mu, 2.0, 50)

    # 先验: μ ~ N(0, 10²), 似然: x_i ~ N(μ, 2²)
    def log_posterior(mu):
        # log prior + log likelihood
        log_prior = -0.5 * ((mu - 0) / 10) ** 2
        log_likelihood = np.sum(-0.5 * ((data - mu) / 2) ** 2)
        return log_prior + log_likelihood

    result = metropolis_hastings(log_posterior, proposal_std=0.5,
                                 n_samples=8000, burn_in=2000, ndim=1)

    print(f"\n[MCMC 正态均值] 真实 μ={true_mu}")
    print(f"  后验均值: {result['mean'][0]:.4f}")
    print(f"  后验标准差: {result['std'][0]:.4f}")
    print(f"  95% CI: [{result['ci_95'][0,0]:.4f}, {result['ci_95'][1,0]:.4f}]")
    print(f"  接受率: {result['acceptance_rate']:.2%}")

    # --- 示例2: 共轭 Beta-Binomial ---
    print(f"\n[Beta-Binomial 共轭]")
    result_beta = bayesian_update(
        prior_params={'type': 'beta_binomial', 'alpha': 2, 'beta': 2},
        likelihood_func=None,
        data={'successes': 7, 'n': 10}
    )
    print(f"  先验: Beta(2,2), 数据: 7/10 成功")
    print(f"  后验: Beta({result_beta['alpha']}, {result_beta['beta']})")
    print(f"  后验均值: {result_beta['posterior_mean']:.4f}")

    # --- 示例3: 药物效果 A/B 测试 ---
    print(f"\n[A/B 测试 - 贝叶斯]")
    # 对照组: Beta(1,1) 先验, 10次转化, 60次展示
    # 实验组: Beta(1,1) 先验, 18次转化, 60次展示
    control = bayesian_update(
        {'type': 'beta_binomial', 'alpha': 1, 'beta': 1},
        None, {'successes': 10, 'n': 60})
    treatment = bayesian_update(
        {'type': 'beta_binomial', 'alpha': 1, 'beta': 1},
        None, {'successes': 18, 'n': 60})

    # 蒙特卡洛比较
    control_samples = np.random.beta(control['alpha'], control['beta'], 100000)
    treatment_samples = np.random.beta(treatment['alpha'], treatment['beta'], 100000)
    prob_treatment_better = np.mean(treatment_samples > control_samples)

    print(f"  对照组后验: Beta({control['alpha']},{control['beta']}), 转化率≈{control['posterior_mean']:.3f}")
    print(f"  实验组后验: Beta({treatment['alpha']},{treatment['beta']}), 转化率≈{treatment['posterior_mean']:.3f}")
    print(f"  P(实验组 > 对照组) = {prob_treatment_better:.2%}")
    if prob_treatment_better > 0.95:
        print("  → 有充分证据表明实验组优于对照组")
