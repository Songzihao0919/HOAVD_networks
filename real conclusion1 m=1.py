import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import math


class DynamicNetworkAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.nodes = set()
        self.time_windows = None
        self.a_i = {}  # Node activity
        self.b_i = {}  # Node vulnerability
        self.edge_ratios = []  # Store ratios for each edge
        self.edge_ratios_squared = []  # Store squared ratios for each edge

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
        print(f"Time steps range: {min_time} - {max_time}, total {len(self.time_windows)} time steps")

    def calculate_activity(self):
        """Calculate node activity"""
        # Track node presence in time steps
        node_appearance = defaultdict(set)
        for _, row in self.df.iterrows():
            node_appearance[row['node1']].add(row['time_window'])
            node_appearance[row['node2']].add(row['time_window'])

        # Calculate activity (appearance count / total time steps)
        total_steps = len(self.time_windows)
        for node in self.nodes:
            active_steps = len(node_appearance.get(node, set()))
            self.a_i[node] = active_steps / total_steps

        print("Node activity calculation completed")

    def calculate_vulnerability(self, inactive_threshold=1000):
        """Calculate node vulnerability"""
        # Create node presence matrix
        node_presence = defaultdict(lambda: defaultdict(bool))
        for _, row in self.df.iterrows():
            node_presence[row['node1']][row['time_window']] = True
            node_presence[row['node2']][row['time_window']] = True

        # Calculate vulnerability (interruption count / (total steps - threshold))
        total_steps = len(self.time_windows)
        valid_periods = total_steps - inactive_threshold

        for node in self.nodes:
            interruption_count = 0
            # Check from threshold time step
            for t in range(self.time_windows[0] + inactive_threshold, self.time_windows[-1] + 1):
                # Check if node is active at current time
                if not node_presence[node].get(t, False):
                    continue

                # Check if inactive for next threshold steps
                inactive_period = True
                for future_t in range(t + 1, t + inactive_threshold + 1):
                    if future_t in node_presence[node] and node_presence[node][future_t]:
                        inactive_period = False
                        break

                if inactive_period:
                    interruption_count += 1

            self.b_i[node] = interruption_count / valid_periods if valid_periods > 0 else 0

        print("Node vulnerability calculation completed")

    def calculate_theoretical_percolation(self):
        """Calculate theoretical percolation threshold based on the formula"""
        # For each edge in the dataset, calculate the ratio
        for _, row in self.df.iterrows():
            node1, node2 = row['node1'], row['node2']

            # Calculate numerator: sum of a_i for nodes in edge
            numerator = self.a_i[node1] + self.a_i[node2]

            # Calculate denominator: sum of (a_i + b_i) for nodes in edge
            denominator = (self.a_i[node1] + self.b_i[node1]) + (self.a_i[node2] + self.b_i[node2])

            # Avoid division by zero
            if denominator > 0:
                ratio = numerator / denominator
                self.edge_ratios.append(ratio)
                self.edge_ratios_squared.append(ratio ** 2)

        # Calculate expectations
        if len(self.edge_ratios) > 0:
            E_ratio = np.mean(self.edge_ratios)
            E_ratio_squared = np.mean(self.edge_ratios_squared)

            # Calculate left side of the equation
            left_side = E_ratio_squared / E_ratio

            # Calculate right side of the equation (for m=1)
            n = len(self.nodes)
            right_side = math.factorial(0) / n  # (m-1)! / n^m, where m=1

            return left_side, right_side, E_ratio, E_ratio_squared
        else:
            return 0, 0, 0, 0

    def simulate_network(self, T, model_type='HOAVD'):
        """
        Simulate network evolution
        model_type: 'HOAVD' or 'HOAD'
        T: Number of time steps to simulate
        """
        # Initialize active edges
        active_edges = set()
        # Group by time window
        grouped = self.df.groupby('time_window')

        # Simulate first T time steps
        for t in self.time_windows[:T]:
            # Get all edges in current time step
            if t not in grouped.groups:
                current_edges = []
            else:
                current_edges = grouped.get_group(t)[['node1', 'node2']].values

            # HOAVD model: Remove edges first
            if model_type == 'HOAVD':
                # Create copy to avoid modification issues
                for edge in list(active_edges):
                    node1, node2 = edge
                    # Calculate average vulnerability as removal probability
                    remove_prob = (self.b_i[node1] + self.b_i[node2]) / 2
                    if np.random.rand() < remove_prob:
                        active_edges.remove(edge)

            # Add edges for current time step
            for edge in current_edges:
                node1, node2 = sorted(edge)  # Sort for consistent representation
                edge_tuple = (node1, node2)

                # Calculate average activity as addition probability
                add_prob = (self.a_i[node1] + self.a_i[node2]) / 2
                if np.random.rand() < add_prob:
                    active_edges.add(edge_tuple)

        return active_edges

    def calculate_network_metrics(self, active_edges):
        """Calculate network metrics"""
        # Create graph
        G = nx.Graph()
        for edge in active_edges:
            G.add_edge(edge[0], edge[1])

        # Return zeros if no nodes
        if G.number_of_nodes() == 0:
            return 0, 0

        # Calculate average degree
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees)

        # Calculate largest connected component relative size
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc) / len(self.nodes)

        return avg_degree, largest_cc_size

    def validate_conclusions(self, T_short=50, T_long=5000):
        """Validate theoretical conclusions"""
        # Get total nodes
        n = len(self.nodes)

        # Calculate theoretical percolation threshold
        left_side, right_side, E_ratio, E_ratio_squared = self.calculate_theoretical_percolation()

        print("\n=== Theoretical Percolation Threshold Calculation ===")
        print(f"E[ratio] = {E_ratio:.6f}")
        print(f"E[ratio^2] = {E_ratio_squared:.6f}")
        print(f"Left side (E[ratio^2] / E[ratio]) = {left_side:.6f}")
        print(f"Right side ((m-1)! / n^m) = {right_side:.6f}")

        # Check if theoretical percolation condition is satisfied
        if left_side > right_side:
            print("✅ Theoretical percolation condition satisfied: Left side > Right side")
            theoretical_percolation = True
        else:
            print("❌ Theoretical percolation condition not satisfied: Left side ≤ Right side")
            theoretical_percolation = False

        # Short timescale validation (T << n)
        print(f"\n=== Short timescale validation (T={T_short} << n={n}) ===")
        edges_hoavd_short = self.simulate_network(T_short, 'HOAVD')
        edges_hoad_short = self.simulate_network(T_short, 'HOAD')

        avg_deg_hoavd_short, cc_hoavd_short = self.calculate_network_metrics(edges_hoavd_short)
        avg_deg_hoad_short, cc_hoad_short = self.calculate_network_metrics(edges_hoad_short)

        print(f"HOAVD model: Avg degree={avg_deg_hoavd_short:.4f}, Largest CC={cc_hoavd_short:.4f}")
        print(f"HOAD model: Avg degree={avg_deg_hoad_short:.4f}, Largest CC={cc_hoad_short:.4f}")

        # Long timescale validation (T >> n)
        print(f"\n=== Long timescale validation (T={T_long} >> n={n}) ===")
        edges_hoavd_long = self.simulate_network(T_long, 'HOAVD')
        edges_hoad_long = self.simulate_network(T_long, 'HOAD')

        avg_deg_hoavd_long, cc_hoavd_long = self.calculate_network_metrics(edges_hoavd_long)
        avg_deg_hoad_long, cc_hoad_long = self.calculate_network_metrics(edges_hoad_long)

        print(f"HOAVD model: Avg degree={avg_deg_hoavd_long:.4f}, Largest CC={cc_hoavd_long:.4f}")
        print(f"HOAD model: Avg degree={avg_deg_hoad_long:.4f}, Largest CC={cc_hoad_long:.4f}")

        # Validate conclusions
        print("\n=== Conclusion Validation ===")

        # Conclusion 1: Vulnerability negligible at T << n
        cc_diff_short = abs(cc_hoavd_short - cc_hoad_short)/cc_hoavd_short
        if cc_diff_short < 0.1:
            print(f"✅ Conclusion 1 verified: At T={T_short}<<n={n}, difference in CC size ({cc_diff_short:.4f}) < 0.1")
        else:
            print(f"⚠️ Conclusion 1 not verified: Significant difference ({cc_diff_short:.4f}) at short timescale")

        # Conclusion 2: Percolation at T >> n
        print(f"\nPercolation threshold (Largest CC > 0.05):")
        print(f"HOAVD model: {cc_hoavd_long:.4f} {'✅' if cc_hoavd_long > 0.05 else '❌'}")
        print(f"HOAD model: {cc_hoad_long:.4f} {'✅' if cc_hoad_long > 0.05 else '❌'}")

        # Compare theoretical prediction with simulation results
        print(f"\n=== Theoretical vs. Simulation Comparison ===")
        if theoretical_percolation and cc_hoavd_long > 0.05:
            print("✅ Theory and simulation agree: Percolation predicted and observed")
        elif not theoretical_percolation and cc_hoavd_long <= 0.05:
            print("✅ Theory and simulation agree: No percolation predicted and observed")
        else:
            print("⚠️ Theory and simulation disagree: Check model assumptions")

        # Visualization
        self.visualize_results(
            cc_hoavd_short, cc_hoad_short, cc_hoavd_long, cc_hoad_long,
            avg_deg_hoavd_short, avg_deg_hoad_short, avg_deg_hoavd_long, avg_deg_hoad_long,
            T_short, T_long, left_side, right_side
        )

    def visualize_results(self, cc_hoavd_short, cc_hoad_short, cc_hoavd_long, cc_hoad_long,
                          avg_deg_hoavd_short, avg_deg_hoad_short, avg_deg_hoavd_long, avg_deg_hoad_long,
                          T_short, T_long, left_side, right_side):
        """Visualize results"""
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Short timescale CC comparison
        ax[0, 0].bar(['HOAVD', 'HOAD'], [cc_hoavd_short, cc_hoad_short], color=['blue', 'orange'])
        ax[0, 0].set_title(f'Short Timescale (T={T_short}<<n^1={n})')
        ax[0, 0].set_ylabel('Largest CC Size')
        ax[0, 0].set_ylim(0, 1)

        # Short timescale degree comparison
        ax[0, 1].bar(['HOAVD', 'HOAD'], [avg_deg_hoavd_short, avg_deg_hoad_short], color=['blue', 'orange'])
        ax[0, 1].set_title(f'Short Timescale (T={T_short}<<n^1={n})')
        ax[0, 1].set_ylabel('Average Degree')

        # Long timescale CC comparison
        ax[1, 0].bar(['HOAVD', 'HOAD'], [cc_hoavd_long, cc_hoad_long], color=['blue', 'orange'])
        ax[1, 0].set_title(f'Long Timescale (T={T_long}>>n^1={n})')
        ax[1, 0].set_ylabel('Largest CC Size')
        ax[1, 0].set_ylim(0, 1)

        # Long timescale degree comparison
        ax[1, 1].bar(['HOAVD', 'HOAD'], [avg_deg_hoavd_long, avg_deg_hoad_long], color=['blue', 'orange'])
        ax[1, 1].set_title(f'Long Timescale (T={T_long}>>n^1={n})')
        ax[1, 1].set_ylabel('Average Degree')

        plt.tight_layout()
        plt.savefig('conclusion_validation.png')

        # Create a separate figure for theoretical results
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.bar(['Left Side', 'Right Side'], [left_side, right_side], color=['green', 'red'])
        ax2.set_title('Theoretical Percolation Threshold')
        ax2.set_ylabel('Value')
        ax2.text(0, left_side, f'{left_side:.6f}', ha='center', va='bottom')
        ax2.text(1, right_side, f'{right_side:.6f}', ha='center', va='bottom')
        plt.savefig('theoretical_percolation.png')

        print("\nVisualizations saved to 'conclusion_validation.png' and 'theoretical_percolation.png'")


# Main program
if __name__ == "__main__":
    # Set file path
    dataset_path = "tij_SFHH.dat_.csv"

    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} does not exist")
        exit(1)

    print("=== Dynamic Network Analyzer ===")
    print(f"Loading dataset: {dataset_path}")

    # Create analyzer instance
    analyzer = DynamicNetworkAnalyzer(dataset_path)

    # Step 1: Load and prepare data
    analyzer.load_and_prepare_data()

    # Step 2: Calculate node properties
    analyzer.calculate_activity()
    analyzer.calculate_vulnerability()

    # Step 3: Validate conclusions
    n = len(analyzer.nodes)
    T_short = 50  # T << n
    T_long = 10000  # T >> n

    print(f"\nNumber of nodes n = {n}")
    print(f"Short timescale: T_short = {T_short} (T << n)")
    print(f"Long timescale: T_long = {T_long} (T >> n)")

    analyzer.validate_conclusions(T_short, T_long)

    print("\n=== Analysis Complete ===")