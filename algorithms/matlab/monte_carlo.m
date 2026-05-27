function result = monte_carlo_sim(problem_type, varargin)
% 蒙特卡洛模拟 — MATLAB 实现
%
% 支持: 'pi' π估计 / 'integrate' 定积分 / 'queue' 排队模拟 / 'risk' 风险分析
%
% 用法:
%   r = monte_carlo_sim('pi', 'n', 50000);
%   r = monte_carlo_sim('integrate', 'func', @(x) sin(x), 'a', 0, 'b', pi, 'n', 100000);

p = inputParser;
addRequired(p, 'problem_type');
addParameter(p, 'n', 50000, @isnumeric);
addParameter(p, 'func', @(x) x);
addParameter(p, 'a', 0, @isnumeric);
addParameter(p, 'b', 1, @isnumeric);
addParameter(p, 'arrival_rate', 5, @isnumeric);
addParameter(p, 'service_rate', 8, @isnumeric);
addParameter(p, 'num_customers', 5000, @isnumeric);
parse(p, problem_type, varargin{:});
opts = p.Results;

rng(42);

switch lower(problem_type)
    case 'pi'
        points = rand(opts.n, 2);
        inside = sum(sum(points .^ 2, 2) <= 1);
        pi_est = 4 * inside / opts.n;
        result.pi_estimate = pi_est;
        result.error = abs(pi_est - pi);

        fprintf('========== 蒙特卡洛 π 估计 ==========\n');
        fprintf('估计值: %.6f\n', pi_est);
        fprintf('误差: %.6f\n', abs(pi_est - pi));

    case 'integrate'
        x = opts.a + (opts.b - opts.a) * rand(opts.n, 1);
        y = opts.func(x);
        mean_y = mean(y);
        std_y = std(y);
        integral_val = (opts.b - opts.a) * mean_y;
        std_err = (opts.b - opts.a) * std_y / sqrt(opts.n);

        result.integral = integral_val;
        result.std_err = std_err;

        fprintf('========== 蒙特卡洛定积分 ==========\n');
        fprintf('积分值: %.6f\n', integral_val);
        fprintf('标准误差: %.6f\n', std_err);

    case 'queue'
        lambda = opts.arrival_rate;
        mu = opts.service_rate;
        n_cust = opts.num_customers;

        inter_arrival = exprnd(1/lambda, n_cust, 1);
        arrival_times = cumsum(inter_arrival);
        service_times = exprnd(1/mu, n_cust, 1);

        wait_times = zeros(n_cust, 1);
        departure_times = zeros(n_cust, 1);
        departure_times(1) = arrival_times(1) + service_times(1);

        for i = 2:n_cust
            start_service = max(arrival_times(i), departure_times(i-1));
            wait_times(i) = start_service - arrival_times(i);
            departure_times(i) = start_service + service_times(i);
        end

        result.avg_wait = mean(wait_times);
        result.max_wait = max(wait_times);
        result.utilization = sum(service_times) / max(departure_times(end), arrival_times(end));
        result.theoretical_avg_wait = lambda / (mu * (mu - lambda));
        result.theoretical_utilization = lambda / mu;

        fprintf('========== M/M/1 排队模拟 (λ=%.0f, μ=%.0f) ==========\n', lambda, mu);
        fprintf('平均等待时间: %.4f (理论: %.4f)\n', result.avg_wait, result.theoretical_avg_wait);
        fprintf('利用率: %.3f (理论: %.3f)\n', result.utilization, result.theoretical_utilization);

    case 'risk'
        % 示例：5年期项目现金流风险
        cf_means = [-100, 30, 35, 40, 45];
        cf_stds = [5, 8, 10, 12, 15];
        rate = 0.08;

        T = length(cf_means);
        npvs = zeros(opts.n, 1);
        t_values = 0:(T-1);

        for t = 1:T
            sim_cf = cf_means(t) + cf_stds(t) * randn(opts.n, 1);
            npvs = npvs + sim_cf / (1 + rate)^t_values(t);
        end

        result.mean_npv = mean(npvs);
        result.std_npv = std(npvs);
        result.VaR_95 = prctile(npvs, 5);
        result.prob_loss = mean(npvs < 0);

        fprintf('========== 投资风险评估 ==========\n');
        fprintf('NPV 均值: %.1f\n', result.mean_npv);
        fprintf('NPV 标准差: %.1f\n', result.std_npv);
        fprintf('VaR (95%%): %.1f\n', result.VaR_95);
        fprintf('亏损概率: %.1f%%\n', result.prob_loss * 100);

    otherwise
        error('Unknown problem_type: %s', problem_type);
end

end
