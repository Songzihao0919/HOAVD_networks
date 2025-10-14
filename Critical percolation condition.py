import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import networkx as nx
from collections import defaultdict
import scipy.optimize as opt


class AccurateCriticalPointModel:
    def __init__(self, n=100000, m_order=3, distribution_type='powerlaw', gamma=2.5):
        self.n = n
        self.m_order = m_order
        self.m_nodes = m_order + 1
        self.dist_type = distribution_type
        self.gamma = gamma
        self.theory_threshold = None
        self.K_value = None
        self.critical_gc_threshold = 0.05  # 巨连通分量阈值定义为5%

    def generate_node_attributes(self):
        """生成节点属性：活跃度a和脆弱度b"""
        if self.dist_type == 'uniform':
            self.a = np.random.uniform(0.01, 1.0, self.n)
            self.b = np.random.uniform(0.01, 1.0, self.n)
        elif self.dist_type == 'powerlaw':
            uniform = np.random.uniform(0, 1, self.n)
            self.a = 0.01 * (1 - uniform) ** (1 / (1 - self.gamma))
            self.a = np.clip(self.a, 0.01, 1.0)
            uniform = np.random.uniform(0, 1, self.n)
            self.b = 0.01 * (1 - uniform) ** (1 / (1 - self.gamma))
            self.b = np.clip(self.b, 0.01, 1.0)

        self._precompute_K_value()
        # 核心理论公式: c* = (1/K) * (m-1)! / n^m
        self.theory_threshold = (1 / self.K_value) * math.factorial(self.m_order - 1) / (self.n ** self.m_order)

    def _precompute_K_value(self, num_samples=500000):
        """预计算K值: K = E[W²] / E[W]"""
        samples = np.random.choice(self.n, size=(num_samples, self.m_nodes))
        sum_a = np.sum(self.a[samples], axis=1)
        sum_b = np.sum(self.b[samples], axis=1)
        W = sum_a / (sum_a + sum_b + 1e-15)
        self.K_value = np.mean(W ** 2) / np.mean(W)

    def steady_state_probability(self, nodes):
        """稳态存在概率: P_∞ = ∑a_i / (∑a_i + ∑b_i)"""
        sum_a = np.sum(self.a[nodes])
        sum_b = np.sum(self.b[nodes])
        return sum_a / (sum_a + sum_b + 1e-15)

    def generate_hyperedges(self, c):
        """根据缩放因子c生成超边集合"""
        # 公式: λ = c * n^{m+1} / (m+1)!
        avg_edges = c * (self.n ** self.m_nodes) / math.factorial(self.m_nodes)
        num_edges = np.random.poisson(avg_edges)
        candidate_edges = np.random.choice(self.n, size=(num_edges, self.m_nodes))
        hyperedges = [
            tuple(sorted(edge))
            for edge in candidate_edges
            if np.random.random() < self.steady_state_probability(edge)
        ]
        return hyperedges

    def giant_component_size(self, hyperedges):
        """使用图投影法计算最大连通分量"""
        if not hyperedges:
            return 0

        graph = nx.Graph()
        graph.add_nodes_from(range(self.n))

        node_to_edges = defaultdict(set)
        for i, edge in enumerate(hyperedges):
            for node in edge:
                node_to_edges[node].add(i)

        for node in range(self.n):
            connected_nodes = set()
            for edge_id in node_to_edges[node]:
                connected_nodes |= set(hyperedges[edge_id])
            connected_nodes.discard(node)

            for neighbor in connected_nodes:
                graph.add_edge(node, neighbor)

        return max((len(comp) for comp in nx.connected_components(graph)), default=0)

    def run_experiment(self, scaling_factors, num_trials=5):
        """记录每个c对应的巨连通分量变化"""
        # 记录每个c的平均巨连通分量
        gc_results = np.zeros(len(scaling_factors))

        total_iterations = num_trials * len(scaling_factors)
        pbar = tqdm(total=total_iterations, desc=f"Percolation (n={self.n}, m={self.m_order})")

        for trial in range(num_trials):
            for idx, c in enumerate(scaling_factors):
                hyperedges = self.generate_hyperedges(c)
                gc_size = self.giant_component_size(hyperedges)
                gc_results[idx] += gc_size / self.n
                pbar.update(1)
                pbar.set_postfix({
                    'trial': trial + 1,
                    'c': f'{c:.2e}',
                    'theory': f'{self.theory_threshold:.2e}'
                })

        pbar.close()

        # 计算平均值
        gc_results /= num_trials

        # 寻找临界点: 使用插值法找到巨连通分量首次达到阈值的精确c值
        exp_critical = None
        for i in range(1, len(scaling_factors)):
            if gc_results[i] > self.critical_gc_threshold:
                # 线性插值法计算精确临界点
                x0, y0 = scaling_factors[i - 1], gc_results[i - 1]
                x1, y1 = scaling_factors[i], gc_results[i]

                # 插值公式: c_crit = x0 + (x1 - x0) * (0.05 - y0) / (y1 - y0)
                exp_critical = x0 + (x1 - x0) * (self.critical_gc_threshold - y0) / (y1 - y0)
                break

        # 如果没有达到阈值，使用最后一点
        if exp_critical is None:
            exp_critical = scaling_factors[-1]

        return {
            'scaling_factors': scaling_factors,
            'theory_threshold': self.theory_threshold,
            'K_value': self.K_value,
            'giant_component': gc_results,
            'experimental_critical': exp_critical
        }


def visualize_results(results, n, m_order, dist_type, gamma=None):
    """可视化结果 - 只显示巨连通分量"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 构建标题
    title_text = f"Steady-state Hypergraph Percolation: n={n}, m={m_order}"
    # 根据分布类型添加描述
    if dist_type == 'powerlaw' and gamma:
        title_text += f", powerlaw dist, γ={gamma}"
    else:
        title_text += f", uniform dist"

    # 设置标题（只调用一次）
    plt.title(title_text, fontsize=14)

    # 巨连通分量曲线
    ax.semilogx(results['scaling_factors'], results['giant_component'], 'g-',
                marker='s', markersize=5, linewidth=2, label='Giant Component')

    # 添加临界点标记
    critical_c = results['experimental_critical']
    critical_index = np.argmin(np.abs(np.array(results['scaling_factors']) - critical_c))
    critical_gc = results['giant_component'][critical_index]

    # 标记临界点 (0.05处)
    ax.plot(critical_c, critical_gc, 'ro', markersize=7,
            label=f'Critical Point')#: {critical_c:.2e}

    # 添加0.05的参考线
    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.7)
    ax.text(max(results['scaling_factors']) * 0.8, 0.05, '5% Threshold',
            verticalalignment='bottom', horizontalalignment='right', color='gray')

    # 理论临界线
    theory_threshold = results['theory_threshold']
    ax.axvline(x=theory_threshold, color='r', linestyle='--',
               linewidth=1.5, label=f'Theory: {theory_threshold:.2e}')

    # 实验临界线
    if 'experimental_critical' in results:
        exp_critical = results['experimental_critical']
        ax.axvline(x=exp_critical, color='m', linestyle='-',
                   linewidth=1.5, label=f'Experiment: {exp_critical:.2e}')

    ax.set_xlabel('Scaling Factor (c)', fontsize=12)
    ax.set_ylabel('Relative Giant Component', fontsize=12, color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_ylim(-0.02, 1.05)

    # 添加K值标注
    #ax.text(0.95, 0.95, f"K = {results['K_value']:.4f}",
    #        transform=ax.transAxes, ha='right', va='top', fontsize=11,
    #        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # 添加实验/理论比率
    if 'experimental_critical' in results:
        ratio = theory_threshold / results['experimental_critical']
        ax.text(0.95, 0.85, f"Theory/Exp Ratio: {ratio:.2f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # 图例放置在左上方
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    dist_label = dist_type + (f"_gamma{gamma}" if gamma else "")
    plt.savefig(f'accurate_percolation_n{n}_m{m_order}_{dist_label}.png', dpi=150)
    plt.show()


def main():
    # 参数配置 - 使用大网络测试
    n = 10000  # 节点数
    m_order = 3  # 超边阶数
    dist_type = 'powerlaw' #'uniform'
    gamma = 2.5

    # 初始化模型 - 专注巨连通分量
    model = AccurateCriticalPointModel(n, m_order, dist_type, gamma)
    model.generate_node_attributes()

    print(f"\n=== 精确临界点超图渗流实验 ===")
    print(f"节点数: {n} | 超边阶数: {m_order}")
    print(f"分布类型: {dist_type} | 幂律指数γ: {gamma}")
    print(f"理论K值: {model.K_value:.4f}")
    print(f"理论临界点: {model.theory_threshold:.2e}")
    print(f"临界巨连通分量阈值: {model.critical_gc_threshold:.2f}")

    # 设置缩放因子扫描范围 - 在临界点附近加密
    theory_threshold = model.theory_threshold
    min_scale = theory_threshold / 10
    max_scale = theory_threshold * 100
    num_points = 50  # 增加点数提高精度
    scaling_factors = np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        num_points
    )

    # 运行实验
    results = model.run_experiment(scaling_factors, num_trials=5)

    # 输出结果
    print(f"\n实验临界点: {results['experimental_critical']:.2e}")
    print(
        f"临界点巨连通分量: {results['giant_component'][np.argmin(np.abs(np.array(scaling_factors) - results['experimental_critical']))]:.4f}")

    # 可视化
    visualize_results(results, n, m_order, dist_type, gamma)


if __name__ == "__main__":
    np.random.seed(42)
    main()