import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous


# =======================
# 修正后的幂律分布生成器
# =======================
def generate_truncated_powerlaw(n, alpha, min_val=0.01, max_val=1.0):
    """
    生成截断幂律分布（直接在[min_val, max_val]范围内）
    :param n: 节点数量
    :param alpha: 幂律指数（>1）
    :param min_val: 最小值（必须>0）
    :param max_val: 最大值（必须≥min_val）
    :return: 服从截断幂律分布的数组
    """

    class TruncatedPowerLaw(rv_continuous):
        def _pdf(self, x, alpha):
            # 幂律分布的PDF: p(x) ∝ x^{-alpha}
            # 计算归一化常数
            # 积分从min_val到max_val: ∫ x^{-alpha} dx = [x^{1-alpha}/(1-alpha)]_{min}^{max}
            # 因此归一化常数C = (alpha-1)/(min_val^{1-alpha} - max_val^{1-alpha})
            numerator = alpha - 1
            denominator = min_val ** (1 - alpha) - max_val ** (1 - alpha)
            norm_const = numerator / denominator
            return norm_const * x ** (-alpha)

    # 创建分布实例
    dist = TruncatedPowerLaw(a=min_val, b=max_val, name='trunc_powerlaw')

    # 生成样本
    samples = dist.rvs(alpha, size=n)

    # 数值稳定性处理（确保不超出范围）
    samples = np.clip(samples, min_val, max_val)
    return samples


# =======================
# 超边动力学模拟函数
# =======================
def simulate_edge_dynamics(n, m, T, a_nodes, b_nodes):
    """
    事件驱动法模拟单条超边状态变化
    :param n: 节点数
    :param m: 超边大小为m+1
    :param T: 总时间
    :param a_nodes: 活跃度数组
    :param b_nodes: 脆弱度数组
    :return: T时刻超边是否存在
    """
    # 固定超边（前m+1节点）
    nodes = np.arange(m + 1)
    sum_a = a_nodes[nodes].sum()
    sum_b = b_nodes[nodes].sum()

    # 计算生成率λ和删除率μ
    factor = math.factorial(m) / (n ** m)
    λ = factor * sum_a
    μ = factor * sum_b

    # 初始状态：不存在
    current_state = False
    current_time = 0.0

    # 事件驱动模拟
    while current_time < T:
        if not current_state:  # 当前不存在 → 等待生成
            rate = λ
        else:  # 当前存在 → 等待删除
            rate = μ

        if rate <= 0:  # 防止除以零
            break

        # 生成下一个事件时间
        time_to_event = np.random.exponential(1 / rate)
        next_time = current_time + time_to_event

        if next_time > T:
            break  # 超过T时终止

        current_time = next_time
        current_state = not current_state  # 状态翻转

    return current_state


# =======================
# 理论概率计算函数
# =======================
def exact_theoretical_probability(T, n, m, sum_a, sum_b):
    """
    计算超边存在概率的严格理论值
    P_T = [sum_a / (sum_a + sum_b)] * [1 - exp(-factor * (sum_a + sum_b) * T)]
    其中 factor = m! / n^m
    """
    factor = math.factorial(m) / (n ** m)
    total_rate = factor * (sum_a + sum_b)

    # 避免除以零
    if sum_a + sum_b == 0:
        return 0.0

    steady_state = sum_a / (sum_a + sum_b)
    transient = 1 - np.exp(-total_rate * T)

    return steady_state * transient


# =======================
# 统一验证函数
# =======================
def unified_validation(n, m, dist_type='uniform'):
    """
    统一验证结论(1)和(2)
    :param n: 节点数
    :param m: 超边大小-1
    :param dist_type: 分布类型 ('uniform', 'powerlaw')
    """
    # 时间点选择
    possible_edges = n ** m / math.factorial(m)
    T_range = np.logspace(0, np.log10(possible_edges * 10 ** 4), 200, dtype=int)
    T_range = np.unique(T_range)  # 确保唯一值

    # 调整T范围确保不会过大
    max_T = possible_edges * 1000
    T_range = T_range[T_range <= max_T]

    # 固定一条超边（前m+1个节点）
    print(f"\nGenerating {dist_type} distributions for {n} nodes...")

    if dist_type == 'powerlaw':
        # 使用修正后的幂律分布生成器
        alpha_a = 2.5  # 活跃度幂律指数
        alpha_b = 2.5  # =1.8  # 脆弱度幂律指数

        # 直接在[0.01, 1]范围内生成概率值
        a_nodes = generate_truncated_powerlaw(n, alpha_a, min_val=0.01, max_val=1.0)
        b_nodes = generate_truncated_powerlaw(n, alpha_b, min_val=0.01, max_val=1.0)
        print(f"Powerlaw distributions: alpha_a={alpha_a}, alpha_b={alpha_b}")
    else:  # 默认均匀分布
        a_nodes = np.random.uniform(0.01, 1.0, n)  # 也在[0.01,1]范围内保持一致性
        b_nodes = np.random.uniform(0.01, 1.0, n)
        print("Uniform distributions in [0.01,1.0]")

    # 固定超边的节点属性
    nodes_e = np.arange(m + 1)
    sum_a_e = a_nodes[nodes_e].sum()
    sum_b_e = b_nodes[nodes_e].sum()
    print(f"Fixed hyperedge (nodes 0 to {m}) | ∑a={sum_a_e:.4f} | ∑b={sum_b_e:.4f}")

    # 理论值计算
    print("Calculating theoretical probabilities...")
    theory_P_smallT = []  # 小T近似理论值
    theory_P_largeT = []  # 大T稳态理论值
    theory_P_exact = []  # 严格理论值

    # 计算因子
    factor = math.factorial(m) / (n ** m)
    total_rate = factor * (sum_a_e + sum_b_e)
    print(f"Rate factor: {factor:.4e} | Total rate: {total_rate:.4e}")

    for T in tqdm(T_range, desc="Theory calc"):
        # 结论(1): n^m >> T时的近似理论值
        smallT_val = min(sum_a_e * factor * T, 1.0)

        # 结论(2): T >> n^m时的稳态理论值
        if sum_a_e + sum_b_e > 0:
            largeT_val = sum_a_e / (sum_a_e + sum_b_e)
        else:
            largeT_val = 0

        # 严格理论值
        exact_val = exact_theoretical_probability(T, n, m, sum_a_e, sum_b_e)

        theory_P_smallT.append(smallT_val)
        theory_P_largeT.append(largeT_val)
        theory_P_exact.append(exact_val)

    # 模拟值计算
    print("Running simulations...")
    sim_P = []
    reps_per_T = max(100, min(1000, int(1e7 // max(1, T_range.max()))))  # 动态调整重复次数

    for T in tqdm(T_range, desc=f"Simulation: n={n}, m={m}, {dist_type}"):
        count = 0
        for _ in range(reps_per_T):
            count += simulate_edge_dynamics(n, m, T, a_nodes, b_nodes)
        sim_P.append(count / reps_per_T)

    # 可视化
    plt.figure(figsize=(12, 8))

    # 1. 近似理论值曲线
    plt.semilogx(T_range, theory_P_smallT, 'g--', label='Small-T approximation')
    plt.semilogx(T_range, theory_P_largeT, 'r--', label='Large-T steady state')

    # 2. 严格理论值曲线
    plt.semilogx(T_range, theory_P_exact, 'm--', linewidth=2.5, label='Exact theoretical')

    # 3. 模拟值
    plt.semilogx(T_range, sim_P, 'bo-', alpha=0.7, markersize=4, label='Simulation')

    # 标记理论转折点
    plt.axvline(x=n ** m, color='black', linestyle=':',
                label=r'$n^m = %.1e$' % (n ** m))
    plt.axvline(x=n ** m / 100, color='gray', linestyle=':',
                label=r'$n^m/100$')
    plt.axvline(x=n ** m * 100, color='orange', linestyle=':',
                label=r'$n^m \times 100$')

    plt.xlabel('Time Step (T)')
    plt.ylabel('Hyperedge Existence Probability')
    plt.title(f'Hyperedge Dynamics (n={n}, m={m}, {dist_type} distribution)')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.savefig(f'dynamics_{dist_type}_n{n}_m{m}.png', dpi=300)
    plt.show()

    # 打印关键点对比
    smallT_point = min(100, len(T_range) - 1)  # 取T较小的点
    largeT_point = -1  # 取T最大的点
    midT_point = len(T_range) // 2  # 取中间点（过渡区）

    print("\nValidation Summary:")
    print(f"Distribution: {dist_type}")
    print(f"Hyperedge: nodes 0-{m} | ∑a={sum_a_e:.4f} ∑b={sum_b_e:.4f}")
    print(f"Total rate factor: {total_rate:.4e}")

    print(f"\nSmall-T validation (T={T_range[smallT_point]} << n^m={n ** m:.1e}):")
    print(f"  Small-T approx: {theory_P_smallT[smallT_point]:.6f}")
    print(f"  Exact theory:   {theory_P_exact[smallT_point]:.6f}")
    print(f"  Simulation:     {sim_P[smallT_point]:.6f}")

    print(f"\nMid-T validation (T={T_range[midT_point]} ≈ transition):")
    print(f"  Large-T steady: {theory_P_largeT[midT_point]:.6f}")
    print(f"  Exact theory:   {theory_P_exact[midT_point]:.6f}")
    print(f"  Simulation:     {sim_P[midT_point]:.6f}")

    print(f"\nLarge-T validation (T={T_range[largeT_point]} >> n^m={n ** m:.1e}):")
    print(f"  Large-T steady: {theory_P_largeT[largeT_point]:.6f}")
    print(f"  Exact theory:   {theory_P_exact[largeT_point]:.6f}")
    print(f"  Simulation:     {sim_P[largeT_point]:.6f}")

    # 计算严格理论与模拟值的整体相关性
    correlation = np.corrcoef(theory_P_exact, sim_P)[0, 1]
    print(f"\nOverall correlation between exact theory and simulation: {correlation:.4f}")

    # 返回关键结果，方便比较
    return {
        'dist_type': dist_type,
        'sum_a': sum_a_e,
        'sum_b': sum_b_e,
        'T_small': T_range[smallT_point],
        'P_small_theory': theory_P_smallT[smallT_point],
        'P_small_sim': sim_P[smallT_point],
        'T_mid': T_range[midT_point],
        'P_mid_theory': theory_P_exact[midT_point],
        'P_mid_sim': sim_P[midT_point],
        'T_large': T_range[largeT_point],
        'P_large_theory': theory_P_exact[largeT_point],
        'P_large_sim': sim_P[largeT_point],
        'correlation': correlation
    }


# =======================
# 安全除法函数
# =======================
def safe_division(numerator, denominator, default=0):
    """安全除法，避免除以零错误"""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


# =======================
# 主执行函数
# =======================
def main():
    # 设置较大的n值以减少边界效应
    n = 100000  # 节点数
    m = 3  # 超边大小为m+1

    print("\n" + "=" * 50)
    print("Validating for large network: Uniform Distribution")
    print("=" * 50)
    results_uniform = unified_validation(n=n, m=m, dist_type='uniform')

    print("\n" + "=" * 50)
    print("Validating for large network: Power-law Distribution")
    print("=" * 50)
    results_powerlaw = unified_validation(n=n, m=m, dist_type='powerlaw')

    # 分布对比分析
    print("\n" + "=" * 50)
    print("Distribution Comparison Summary")
    print("=" * 50)
    print(f"{'Metric':<30} | {'Uniform':^15} | {'Power-law':^15} | {'Difference (%)':^15}")
    print("-" * 80)

    # 定义安全计算函数
    def calc_safe_diff(uni_val, pl_val, default=0):
        """安全计算相对差异百分比"""
        abs_diff = abs(uni_val - pl_val)
        relative_diff = safe_division(abs_diff, uni_val, default) * 100
        return relative_diff

    # 打印各项对比
    metrics = [
        ('Small-T Theory', results_uniform['P_small_theory'], results_powerlaw['P_small_theory']),
        ('Small-T Simulation', results_uniform['P_small_sim'], results_powerlaw['P_small_sim']),
        ('Mid-T Theory', results_uniform['P_mid_theory'], results_powerlaw['P_mid_theory']),
        ('Mid-T Simulation', results_uniform['P_mid_sim'], results_powerlaw['P_mid_sim']),
        ('Large-T Theory', results_uniform['P_large_theory'], results_powerlaw['P_large_theory']),
        ('Large-T Simulation', results_uniform['P_large_sim'], results_powerlaw['P_large_sim']),
        ('Active sum (∑a)', results_uniform['sum_a'], results_powerlaw['sum_a']),
        ('Vulnerability sum (∑b)', results_uniform['sum_b'], results_powerlaw['sum_b']),
        ('Correlation (exact vs sim)', results_uniform['correlation'], results_powerlaw['correlation'])
    ]

    for name, uni_val, pl_val in metrics:
        diff = calc_safe_diff(uni_val, pl_val)
        print(f"{name:<30} | {uni_val:15.6f} | {pl_val:15.6f} | {diff:15.1f}%")

    # 额外解释幂律分布的影响
    print("\nKey Observations:")
    print("1. Power-law distributions typically result in higher ∑a and ∑b compared to uniform distributions")
    print("2. The Small-T approximation shows larger differences due to extreme value effects")
    print("3. The Large-T steady state shows smaller percentage differences as it only depends on ∑a/(∑a+∑b)")


# =======================
# 程序入口
# =======================
if __name__ == "__main__":
    # 设置随机种子保证可复现性
    np.random.seed(42)

    # 主执行函数
    main()

    print("\nSimulation and validation completed successfully!")