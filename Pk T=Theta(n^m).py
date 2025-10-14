import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import gaussian_kde
from tqdm import tqdm
import time
import math

# 1. 生成标准化节点属性
def generate_node_attributes(n, alpha=2.5, min_val=0.01, max_val=1.0):
    """生成标准化节点属性"""
    # 生成幂律分布样本
    u = np.random.rand(n)
    xmin, xmax = min_val, max_val
    exponent = 1 - alpha
    a0 = (xmin ** exponent - u * (xmin ** exponent - xmax ** exponent)) ** (1 / exponent)

    u = np.random.rand(n)
    b0 = (xmin ** exponent - u * (xmin ** exponent - xmax ** exponent)) ** (1 / exponent)

    # 标准化 (a + b = 1)
    total = a0 + b0
    a = a0 / total
    b = b0 / total
    return a, b


# 2. 计算理论度分布
def theoretical_degree_distribution(a, b, n, m, k_values):
    """计算理论度分布"""
    N_e = comb(n - 1, m) #comb(n - 1, m)  # 每个节点的潜在连接数
    a_mean = np.mean(a)  # <a>

    # 计算标准化后的a,b的密度估计
    kde_a = gaussian_kde(a)
    kde_b = gaussian_kde(b)

    C = 1 - math.exp(-math.factorial(m + 1))

    # 计算理论P(k)
    p_k = []
    for k in k_values:
        q = k / N_e
        t_val = (1 + m) * q / C - m * a_mean
        # 确保t在合理范围内
        if t_val < 0 or t_val > 1:
            p_k.append(0)
        else:
            rho_t_1 = kde_a(t_val)[0]
            rho_t_2 = kde_b(1-t_val)[0]
            denominator = N_e * np.sqrt(C ** 2 - 2 * C * q + 2 * q ** 2 )
            if denominator > 1e-6:  # 避免除以0
                p_k.append((1+m) * rho_t_1 * rho_t_2/ denominator)
            else:
                p_k.append(0)

    return np.array(p_k)


# 3. 高效模拟稳态网络
def simulate_hypergraph(a, b, n, m):
    """高效模拟稳态网络"""
    # 计算每个节点的期望度
    expected_degrees = np.zeros(n)

    # 预计算所有可能超边的概率太慢，改用蒙特卡洛方法
    # 使用重要性抽样：生成随机超边并加权
    num_samples = min(10000000, int(comb(n, m + 1)))  #comb(n, m + 1) # 限制样本数量
    degrees = np.zeros(n)

    C = 1 - math.exp(-math.factorial(m + 1))

    for _ in range(num_samples):
        # 随机选择m+1个节点
        nodes = np.random.choice(n, m + 1, replace=False)
        sum_a = np.sum(a[nodes])
        p_e = sum_a * C / (m + 1)  # 超边存在概率

        # 根据概率决定是否添加超边
        if np.random.rand() < p_e:
            degrees[nodes] += 1

    # 按采样比例缩放度值
    scaling_factor = comb(n, m + 1) / num_samples #comb(n, m + 1)
    return degrees * scaling_factor


# 4. 计算模拟度分布
def simulate_degree_distribution(a, b, n, m, k_values):
    """计算模拟度分布"""
    # 模拟网络
    degrees = simulate_hypergraph(a, b, n, m)

    # 计算概率密度
    kde = gaussian_kde(degrees)
    return kde(k_values)


# 主函数
def main():
    # 参数设置
    n = 10000  # 节点数（为了计算效率适当减小）
    m_values = [1, 2, 5]  #[1, 2, 5] # 超边阶数
    alpha = 2.5  # 幂律分布参数
    min_val, max_val = 0.01, 1.0  # 取值范围

    # 生成节点属性
    a, b = generate_node_attributes(n, alpha, min_val, max_val)

    # 创建图形
    fig, axes = plt.subplots(1, len(m_values), figsize=(18, 5))
    if len(m_values) == 1:
        axes = [axes]

    for i, m in enumerate(m_values):
        print(f"\nProcessing m = {m}...")
        start_time = time.time()

        # 计算理论度分布
        k_min = 0
        k_max = int(comb(n - 1, m))  #comb(n - 1, m) # 最大可能度
        k_values = np.linspace(k_min, k_max, 500)

        p_k_theory = theoretical_degree_distribution(a, b, n, m, k_values)

        # 计算模拟度分布
        p_k_sim = simulate_degree_distribution(a, b, n, m, k_values)

        # 归一化
        p_k_theory = p_k_theory / np.max(p_k_theory)
        p_k_sim = p_k_sim / np.max(p_k_sim)

        # 绘图
        ax = axes[i]
        ax.plot(k_values, p_k_theory, 'b-', linewidth=2, label='Theoretical')
        ax.plot(k_values, p_k_sim, 'r--', linewidth=2, label='Simulated')
        ax.set_xlabel('Degree k')
        ax.set_ylabel('Normalized P(k)')
        ax.set_title(f'm = {m}, n = {n}')
        ax.legend()
        ax.grid(True)

        print(f"Completed m={m} in {time.time() - start_time:.2f} seconds")

    plt.tight_layout()
    plt.savefig('degree_distribution_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()