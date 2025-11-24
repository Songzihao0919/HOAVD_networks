import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import networkx as nx
from itertools import combinations


class RealDataHyperedgeDynamicsM3:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.nodes = set()
        self.time_windows = None
        self.a_i = {}  # Node activity for m=3
        self.b_i = {}  # Node vulnerability for m=3
        self.hyper_edges = defaultdict(list)  # Store hyperedges for each time window
        self.original_steps = None

    def load_and_prepare_data(self):
        """Load dataset and prepare analysis environment"""
        # Load CSV file
        self.df = pd.read_csv(self.file_path)

        # Extract all nodes
        self.nodes = set(self.df['node1']).union(set(self.df['node2']))
        print(f"Dataset contains {len(self.nodes)} unique nodes")

        # Determine all time windows
        min_time = self.df['time_window'].min()
        max_time = self.df['time_window'].max()
        self.time_windows = list(range(min_time, max_time + 1))
        self.original_steps = len(self.time_windows)
        print(f"Time steps range: {min_time} - {max_time}, total {self.original_steps} time steps")

    def identify_4node_cliques(self):
        """Identify 4-node cliques (m=3 hyperedges) in each time window"""
        print("Identifying 4-node cliques for m=3...")

        # Group by time window
        grouped = self.df.groupby('time_window')

        for time_win, group in tqdm(grouped, total=len(grouped), desc="Processing time windows"):
            # Create graph for this time window
            G = nx.Graph()
            for _, row in group.iterrows():
                G.add_edge(row['node1'], row['node2'])

            # Find all cliques of size 4 (4-node complete subgraphs)
            cliques = list(nx.find_cliques(G))
            four_cliques = [tuple(sorted(clique)) for clique in cliques if len(clique) == 4]

            # Store unique 4-node cliques
            self.hyper_edges[time_win] = list(set(four_cliques))

        # Add empty time windows for completeness
        for time_win in self.time_windows:
            if time_win not in self.hyper_edges:
                self.hyper_edges[time_win] = []

        total_hyperedges = sum(len(v) for v in self.hyper_edges.values())
        print(f"Identified {total_hyperedges} 4-node cliques (m=3)")

    def calculate_activity_vulnerability_m3(self):
        """Calculate node activity and vulnerability for m=3 hyperedges"""
        print("Calculating activity and vulnerability for m=3...")

        # Track node presence in 4-node cliques
        node_clique_windows = defaultdict(set)
        node_active_windows = defaultdict(list)

        # First pass: find all time windows where each node appears in a 4-clique
        for time_win, cliques in self.hyper_edges.items():
            for clique in cliques:
                for node in clique:
                    node_clique_windows[node].add(time_win)
                    node_active_windows[node].append(time_win)

        # Calculate activity: proportion of time windows with clique participation
        total_steps = len(self.time_windows)
        for node in self.nodes:
            active_count = len(node_clique_windows.get(node, set()))
            self.a_i[node] = active_count / total_steps

        # Calculate vulnerability: interruption frequency
        inactive_threshold = 1000
        for node in tqdm(self.nodes, desc="Calculating vulnerability"):
            active_windows = sorted(node_active_windows[node])
            interruption_count = 0

            for i in range(len(active_windows)):
                current_window = active_windows[i]

                # Only consider events that have enough future windows
                if current_window <= max(self.time_windows) - inactive_threshold:
                    # Check if there's another clique within the threshold
                    found_next = False
                    for j in range(i + 1, len(active_windows)):
                        if active_windows[j] <= current_window + inactive_threshold:
                            found_next = True
                            break

                    # If no clique found within threshold, count as interruption
                    if not found_next:
                        interruption_count += 1

            # Calculate vulnerability
            valid_periods = total_steps - inactive_threshold
            self.b_i[node] = interruption_count / valid_periods if valid_periods > 0 else 0

        print("Activity and vulnerability calculation for m=3 completed")

    def find_most_frequent_hyperedge(self):
        """Find the most frequent hyperedge (4-node clique) in the dataset"""
        hyperedge_counts = defaultdict(int)

        for time_win, cliques in self.hyper_edges.items():
            for clique in cliques:
                hyperedge_counts[clique] += 1

        if hyperedge_counts:
            most_frequent = max(hyperedge_counts.items(), key=lambda x: x[1])
            return most_frequent[0], most_frequent[1]
        else:
            # Fallback: use first four nodes
            nodes_list = list(self.nodes)[:4]
            return tuple(nodes_list), 0

    def simulate_hyperedge_dynamics(self, n, m, T, hyperedge_nodes):
        """
        Event-driven simulation of hyperedge state changes
        :param n: number of nodes
        :param m: hyperedge order (m=3 for 4-node cliques)
        :param T: total time
        :param hyperedge_nodes: tuple of nodes in the hyperedge
        :return: hyperedge existence at time T
        """
        # Calculate sum of activities and vulnerabilities
        sum_a = sum(self.a_i.get(node, 0) for node in hyperedge_nodes)
        sum_b = sum(self.b_i.get(node, 0) for node in hyperedge_nodes)

        # Calculate formation and dissolution rates
        factor = math.factorial(m) / (n ** m)
        λ = factor * sum_a  # Formation rate
        μ = factor * sum_b  # Dissolution rate

        # Initial state: does not exist
        current_state = False
        current_time = 0.0

        # Event-driven simulation
        while current_time < T:
            if not current_state:  # Current state:不存在 → wait for formation
                rate = λ
            else:  # Current state: exists → wait for dissolution
                rate = μ

            if rate <= 0:  # Prevent division by zero
                break

            # Generate next event time
            time_to_event = np.random.exponential(1 / rate)
            next_time = current_time + time_to_event

            if next_time > T:
                break  # Terminate if exceeds T

            current_time = next_time
            current_state = not current_state  # State flip

        return current_state

    def exact_theoretical_probability(self, T, n, m, hyperedge_nodes):
        """
        Calculate exact theoretical probability of hyperedge existence
        P_T = [sum_a / (sum_a + sum_b)] * [1 - exp(-factor * (sum_a + sum_b) * T)]
        """
        sum_a = sum(self.a_i.get(node, 0) for node in hyperedge_nodes)
        sum_b = sum(self.b_i.get(node, 0) for node in hyperedge_nodes)

        factor = math.factorial(m) / (n ** m)
        total_rate = factor * (sum_a + sum_b)

        if sum_a + sum_b == 0:
            return 0.0

        steady_state = sum_a / (sum_a + sum_b)
        transient = 1 - np.exp(-total_rate * T)

        return steady_state * transient

    def analyze_hyperedge_dynamics_m3(self, num_simulations=500):
        """
        Analyze hyperedge dynamics for m=3 using real data
        """
        n = len(self.nodes)
        m = 3  # Fixed for 4-node clique hyperedges
        print(f"Analyzing hyperedge dynamics for m={m}, n={n}")

        # Find the most frequent hyperedge
        hyperedge_nodes, frequency = self.find_most_frequent_hyperedge()
        print(f"Selected most frequent hyperedge: {hyperedge_nodes} (appears {frequency} times)")

        # Calculate properties of selected hyperedge
        sum_a = sum(self.a_i.get(node, 0) for node in hyperedge_nodes)
        sum_b = sum(self.b_i.get(node, 0) for node in hyperedge_nodes)
        print(f"Hyperedge properties: ∑a={sum_a:.6f}, ∑b={sum_b:.6f}")

        # Time range setup - adjust for m=3 (n^3 is much larger)
        possible_hyperedges = n ** m / math.factorial(m)
        T_range = np.logspace(0, np.log10(possible_hyperedges * 10 ** 4), 50, dtype=int)
        T_range = np.unique(T_range)
        max_T = possible_hyperedges * 1000
        T_range = T_range[T_range <= max_T]

        print(f"Time range: {T_range.min()} to {T_range.max()} steps")
        print(f"n^{m} = {n ** m:.2e}")

        # Theoretical calculations
        print("Calculating theoretical probabilities...")
        theory_P_smallT = []  # Small T approximation
        theory_P_largeT = []  # Large T steady state
        theory_P_exact = []  # Exact theoretical value

        factor = math.factorial(m) / (n ** m)
        total_rate = factor * (sum_a + sum_b)

        for T in tqdm(T_range, desc="Theory calculation"):
            # Small T approximation (T << n^m)
            smallT_val = min(sum_a * factor * T, 1.0)

            # Large T steady state (T >> n^m)
            if sum_a + sum_b > 0:
                largeT_val = sum_a / (sum_a + sum_b)
            else:
                largeT_val = 0

            # Exact theoretical value
            exact_val = self.exact_theoretical_probability(T, n, m, hyperedge_nodes)

            theory_P_smallT.append(smallT_val)
            theory_P_largeT.append(largeT_val)
            theory_P_exact.append(exact_val)

        # Simulation
        print("Running simulations...")
        sim_P = []
        reps_per_T = max(10, min(100, int(1e5 // max(1, T_range.max()))))

        for T in tqdm(T_range, desc=f"Simulation: m={m}"):
            count = 0
            for _ in range(reps_per_T):
                count += self.simulate_hyperedge_dynamics(n, m, T, hyperedge_nodes)
            sim_P.append(count / reps_per_T)

        # Visualization
        self._plot_results_m3(T_range, theory_P_smallT, theory_P_largeT,
                              theory_P_exact, sim_P, n, m, hyperedge_nodes)

        return {
            'T_range': T_range,
            'theory_smallT': theory_P_smallT,
            'theory_largeT': theory_P_largeT,
            'theory_exact': theory_P_exact,
            'simulation': sim_P,
            'hyperedge_nodes': hyperedge_nodes,
            'sum_a': sum_a,
            'sum_b': sum_b,
            'frequency': frequency
        }

    def _plot_results_m3(self, T_range, theory_P_smallT, theory_P_largeT,
                         theory_P_exact, sim_P, n, m, hyperedge_nodes):
        """Plot results for m=3 similar to Fig.1 in the paper"""
        plt.figure(figsize=(12, 8))

        # Plot theoretical curves
        plt.semilogx(T_range, theory_P_smallT, 'g--', linewidth=2,
                     label='Small-T approximation (T ≪ n^m)')
        plt.semilogx(T_range, theory_P_largeT, 'r--', linewidth=2,
                     label='Large-T steady state (T ≫ n^m)')
        plt.semilogx(T_range, theory_P_exact, 'm-', linewidth=2,
                     label='Exact theoretical value', alpha=0.8)

        # Plot simulation results
        plt.semilogx(T_range, sim_P, 'bo-', alpha=0.7, markersize=4,
                     label='Simulation results')

        # Mark theoretical transition points
        n_power_m = n ** m
        plt.axvline(x=n_power_m, color='black', linestyle=':',
                    label=r'$n^m = %.2e$' % n_power_m)
        plt.axvline(x=n_power_m / 100, color='gray', linestyle=':',
                    label=r'$n^m/100$')
        plt.axvline(x=n_power_m * 100, color='orange', linestyle=':',
                    label=r'$n^m \times 100$')

        plt.xlabel('Time Step (T)', fontsize=12)
        plt.ylabel('Hyperedge Existence Probability', fontsize=12)

        # 应用改进的格式
        node_ids = [int(node) for node in hyperedge_nodes]
        node_labels = ", ".join([f"node {node_id}" for node_id in node_ids])
        plt.title(f'(c) Hyperedge Dynamics with SFHH conference dataset (m={m}, n={n})\n'
                  f'Hyperedge: ({node_labels})', fontsize=14)

        # 曲线图例在左上角
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, which="both", linestyle='--', alpha=0.7)

        # ∑a和∑b文本框在右下角（应用您的改进：0.98, 0.06）
        sum_a = sum(self.a_i.get(node, 0) for node in hyperedge_nodes)
        sum_b = sum(self.b_i.get(node, 0) for node in hyperedge_nodes)
        textstr = f'∑a = {sum_a:.6f}\n∑b = {sum_b:.6f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.98, 0.06, textstr, transform=plt.gca().transAxes,
                       fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                       bbox=props)

        plt.tight_layout()
        plt.savefig(f'real_data_hyperedge_dynamics_m={m}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Plot saved as 'real_data_hyperedge_dynamics_m={m}.png'")


def main():
    """Main function to run the analysis for m=3"""
    # Set file path
    dataset_path = "tij_SFHH.dat_.csv"

    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} does not exist")
        return

    print("=== Real Data Hyperedge Dynamics Analyzer (m=3) ===")
    print(f"Loading dataset: {dataset_path}")

    # Create analyzer instance
    analyzer = RealDataHyperedgeDynamicsM3(dataset_path)

    # Step 1: Load and prepare data
    analyzer.load_and_prepare_data()

    # Step 2: Identify 4-node cliques (m=3 hyperedges)
    analyzer.identify_4node_cliques()

    # Step 3: Calculate activity and vulnerability for m=3
    analyzer.calculate_activity_vulnerability_m3()

    # Step 4: Analyze hyperedge dynamics
    results = analyzer.analyze_hyperedge_dynamics_m3(num_simulations=500)

    # Print summary
    print("\n" + "=" * 50)
    print("Analysis Summary for m=3")
    print("=" * 50)
    print(f"Network size: n = {len(analyzer.nodes)}")
    print(f"Hyperedge order: m = 3")
    print(f"Selected hyperedge: {results['hyperedge_nodes']}")
    print(f"Hyperedge frequency: {results['frequency']} appearances")
    print(f"Sum of activities (∑a): {results['sum_a']:.6f}")
    print(f"Sum of vulnerabilities (∑b): {results['sum_b']:.6f}")

    # Calculate correlation between theory and simulation
    correlation = np.corrcoef(results['theory_exact'], results['simulation'])[0, 1]
    print(f"Correlation between exact theory and simulation: {correlation:.4f}")

    # Find time points for different regimes
    n_power_m = len(analyzer.nodes) ** 3
    smallT_idx = np.argmax(results['T_range'] > n_power_m / 1000)  # T << n^m
    largeT_idx = -1  # T >> n^m
    midT_idx = len(results['T_range']) // 2  # Transition region

    print(f"\nRegime Analysis:")
    print(f"Small-T (T={results['T_range'][smallT_idx]} << n^m={n_power_m:.2e}):")
    print(f"  Theory: {results['theory_exact'][smallT_idx]:.6f}")
    print(f"  Simulation: {results['simulation'][smallT_idx]:.6f}")

    print(f"Large-T (T={results['T_range'][largeT_idx]} >> n^m={n_power_m:.2e}):")
    print(f"  Theory: {results['theory_exact'][largeT_idx]:.6f}")
    print(f"  Simulation: {results['simulation'][largeT_idx]:.6f}")

    # 解释超边选择逻辑
    print("\n" + "=" * 50)
    print("Hyperedge Selection Explanation")
    print("=" * 50)
    print("The selected hyperedge (4-node clique) is chosen based on:")
    print("1. Maximum frequency: The 4-clique that appears most often in the dataset")
    print("2. Statistical significance: High-frequency interactions provide better dynamics")
    print("3. Data-driven selection: Based on actual interaction patterns in the SFHH dataset")
    print("4. For m=3, we identify 4-node cliques (complete subgraphs of size 4)")
    print("5. These represent higher-order interactions where all 4 nodes interact simultaneously")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run main function
    main()