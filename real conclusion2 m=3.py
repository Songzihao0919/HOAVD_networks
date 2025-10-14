import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import itertools
import csv


class HyperedgeConvergenceValidatorM3:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.nodes = set()
        self.time_windows = None
        self.hyper_edges = defaultdict(list)  # Store hyperedges for each time window
        self.a_i = {}  # Node activity for m=3
        self.b_i = {}  # Node vulnerability for m=3
        self.unique_hyperedges = set()  # Store all unique hyperedges
        self.n = 0  # Number of nodes
        self.m = 3  # Order of interaction (m=3 for four-node interactions)

    def load_and_prepare_data(self):
        """Load dataset and prepare analysis environment"""
        # Load CSV file
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset contains {len(self.df)} rows")

        # Convert node IDs to integers
        self.df['node1'] = self.df['node1'].astype(int)
        self.df['node2'] = self.df['node2'].astype(int)

        # Extract all unique nodes
        self.nodes = set(self.df['node1']).union(set(self.df['node2']))
        self.n = len(self.nodes)
        print(f"Dataset contains {self.n} unique nodes")

        # Create continuous time windows (1626 to 7341)
        min_win = self.df['time_window'].min()
        max_win = self.df['time_window'].max()
        self.time_windows = list(range(min_win, max_win + 1))
        total_windows = len(self.time_windows)
        print(f"Total time windows: {total_windows} (from {min_win} to {max_win})")

        # Identify four-node interactions (m=3 hyperedges)
        self.identify_tetrahedrons()

        # Add empty time windows
        for time_win in self.time_windows:
            if time_win not in self.hyper_edges:
                self.hyper_edges[time_win] = []

        total_hyperedges = sum(len(v) for v in self.hyper_edges.values())
        print(f"Identified {total_hyperedges} four-node interactions")

    def identify_tetrahedrons(self):
        """Identify four-node interactions (m=3 hyperedges) in each time window"""
        print("Identifying four-node interactions...")

        # Group by time window
        grouped = self.df.groupby('time_window')

        for time_win, group in tqdm(grouped, total=len(grouped)):
            # Create graph for this time window
            G = nx.Graph()
            for _, row in group.iterrows():
                G.add_edge(row['node1'], row['node2'])

            # Identify four-node cliques
            tetrahedrons = set()
            for clique in nx.find_cliques(G):
                if len(clique) == 4:
                    tetrahedrons.add(tuple(sorted(clique)))

            self.hyper_edges[time_win] = list(tetrahedrons)

            # Add to unique hyperedges
            for tetra in tetrahedrons:
                self.unique_hyperedges.add(tetra)

    def calculate_activity(self):
        """Calculate node activity (a_i)"""
        # Track time windows where each node participates in a four-node interaction
        node_active_windows = defaultdict(set)

        for time_win, tetrahedrons in self.hyper_edges.items():
            for tetra in tetrahedrons:
                for node in tetra:
                    node_active_windows[node].add(time_win)

        # Calculate activity = active windows / total windows
        total_windows = len(self.time_windows)
        for node in self.nodes:
            active_count = len(node_active_windows.get(node, set()))
            self.a_i[node] = active_count / total_windows

        print("Node activity (m=3) calculation completed")

    def calculate_vulnerability(self, inactive_threshold=1000):
        """Calculate node vulnerability (b_i)"""
        # Track interruption count
        interruption_count = defaultdict(int)
        interrupted_windows = set()  # Track already counted interruption windows

        # Find time windows where each node is active in a four-node interaction
        node_active_windows = defaultdict(list)
        for time_win, tetrahedrons in self.hyper_edges.items():
            for tetra in tetrahedrons:
                for node in tetra:
                    node_active_windows[node].append(time_win)

        # Check for interruptions for each node
        for node in tqdm(self.nodes, desc="Calculating vulnerability"):
            active_windows = sorted(node_active_windows[node])
            for i in range(len(active_windows)):
                current_window = active_windows[i]

                # Skip already counted windows
                if current_window in interrupted_windows:
                    continue

                # Only consider events with at least 1000 future windows
                if current_window <= max(self.time_windows) - inactive_threshold:
                    # Check if there's another four-node interaction within next 1000 windows
                    found_next = False
                    for j in range(i + 1, len(active_windows)):
                        if active_windows[j] <= current_window + inactive_threshold:
                            found_next = True
                            break

                    # If no interaction found within next 1000 windows, count as interruption
                    if not found_next:
                        interruption_count[node] += 1
                        interrupted_windows.add(current_window)

        # Calculate vulnerability
        valid_windows = len(self.time_windows) - inactive_threshold
        for node in self.nodes:
            if valid_windows > 0:
                self.b_i[node] = interruption_count[node] / valid_windows
            else:
                self.b_i[node] = 0

        print("Node vulnerability (m=3) calculation completed")

    def simulate_single_hyperedge_dynamics(self, T, a_values, b_values):
        """Simulate dynamics of a single hyperedge"""
        # Calculate correct factor: m! / n^m
        factor = math.factorial(self.m) / (self.n ** self.m)

        # Calculate rates
        λ = factor * sum(a_values)  # Activation rate
        μ = factor * sum(b_values)  # Deactivation rate

        # Initial state: inactive
        current_state = False
        current_time = 0.0

        # Event-driven simulation
        while current_time < T:
            if not current_state:
                # Waiting for activation
                if λ <= 0:
                    break
                time_to_event = np.random.exponential(1 / λ)
                next_time = current_time + time_to_event

                if next_time > T:
                    break

                current_time = next_time
                current_state = True
            else:
                # Waiting for deactivation
                if μ <= 0:
                    break
                time_to_event = np.random.exponential(1 / μ)
                next_time = current_time + time_to_event

                if next_time > T:
                    break

                current_time = next_time
                current_state = False

        return current_state

    def validate_convergence(self, T_max=2 * 10 ** 9):
        """Validate convergence of hyperedge steady-state probabilities"""
        # Select top 4 most frequent hyperedges
        hyperedge_counts = defaultdict(int)
        for time_win, tetrahedrons in self.hyper_edges.items():
            for tetra in tetrahedrons:
                hyperedge_counts[tetra] += 1

        # Sort hyperedges by frequency
        sorted_hyperedges = sorted(hyperedge_counts.items(), key=lambda x: x[1], reverse=True)
        top_hyperedges = [hyperedge for hyperedge, count in sorted_hyperedges[:4]]

        print(f"\nSelected top {len(top_hyperedges)} hyperedges for validation:")
        for i, hyperedge in enumerate(top_hyperedges):
            node1, node2, node3, node4 = hyperedge
            print(
                f"{i + 1}. Hyperedge (nodes {int(node1)}, {int(node2)}, {int(node3)}, {int(node4)}): {hyperedge_counts[hyperedge]} interactions")

        # Create figure with 4 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Steady-State Probability Convergence for selected hyperedges (m=3, T={T_max}>>403^3)', fontsize=16)

        # Validate convergence for each hyperedge
        results = []
        for i, hyperedge in enumerate(top_hyperedges):
            node1, node2, node3, node4 = hyperedge
            a1, a2, a3, a4 = self.a_i[node1], self.a_i[node2], self.a_i[node3], self.a_i[node4]
            b1, b2, b3, b4 = self.b_i[node1], self.b_i[node2], self.b_i[node3], self.b_i[node4]

            # Calculate theoretical steady-state probability
            numerator = a1 + a2 + a3 + a4
            denominator = numerator + b1 + b2 + b3 + b4
            theoretical_prob = numerator / denominator if denominator > 0 else 0

            # Print node properties
            print(f"\nHyperedge (nodes {int(node1)}, {int(node2)}, {int(node3)}, {int(node4)}) properties:")
            print(f"  Node {int(node1)}: a_i = {a1:.6f}, b_i = {b1:.6f}")
            print(f"  Node {int(node2)}: a_i = {a2:.6f}, b_i = {b2:.6f}")
            print(f"  Node {int(node3)}: a_i = {a3:.6f}, b_i = {b3:.6f}")
            print(f"  Node {int(node4)}: a_i = {a4:.6f}, b_i = {b4:.6f}")
            print(f"  ∑a = {a1 + a2 + a3 + a4:.6f}, ∑b = {b1 + b2 + b3 + b4:.6f}")
            print(f"  Theoretical P∞ = {theoretical_prob:.6f}")

            # Set time points
            T_points = np.logspace(0, np.log10(T_max), 50, dtype=int)
            T_points = np.unique(T_points)  # Ensure unique values

            # Store results
            empirical_probs = []
            exact_theoretical = []

            # Simulate for each time point
            for T in tqdm(T_points, desc=f"Simulating hyperedge (nodes {int(node1)}, {int(node2)}, {int(node3)}, {int(node4)})"):
                active_count = 0
                # Increase simulations for better accuracy
                for _ in range(100):  # Reduced for efficiency
                    if self.simulate_single_hyperedge_dynamics(T, [a1, a2, a3, a4], [b1, b2, b3, b4]):
                        active_count += 1
                empirical_probs.append(active_count / 100)

                # Calculate exact theoretical probability
                factor = math.factorial(self.m) / (self.n ** self.m)
                total_rate = factor * (a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4)
                if total_rate > 0:
                    exact_val = theoretical_prob * (1 - math.exp(-total_rate * T))
                else:
                    exact_val = 0
                exact_theoretical.append(exact_val)

            # Select subplot
            row_idx = i // 2
            col_idx = i % 2
            ax = axs[row_idx, col_idx]

            # Plot results
            ax.semilogx(T_points, [theoretical_prob] * len(T_points), 'r--', label='Theoretical Steady State')
            ax.semilogx(T_points, exact_theoretical, 'g-', label='Exact Theoretical')
            ax.semilogx(T_points, empirical_probs, 'bo-', markersize=4, label='Empirical Probability')

            # Set title and labels
            ax.set_title(f'Hyperedge (nodes {int(node1)}, {int(node2)}, {int(node3)}, {int(node4)})', fontsize=12)
            ax.set_xlabel('Time (T)')
            ax.set_ylabel('Hyperedge Probability')
            ax.grid(True, which="both", linestyle='--', alpha=0.7)
            ax.legend()

            # Add annotation
            ax.annotate(f'P∞ = {theoretical_prob:.4f}',
                        xy=(0.95, 0.05), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

            # Store results for summary
            results.append({
                'hyperedge': hyperedge,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_probs[-1]  # Last point at T_max
            })

        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig('hyperedge_convergence_summary_m3.png', dpi=300)
        print("\nVisualization saved to 'hyperedge_convergence_summary_m3.png'")

        # Print summary
        print("\n=== Validation Summary ===")
        print(f"{'Hyperedge':<60} {'Theoretical':<12} {'Empirical':<12}")
        print("-" * 85)
        for result in results:
            node1, node2, node3, node4 = result['hyperedge']
            hyperedge_str = f"(nodes {int(node1)}, {int(node2)}, {int(node3)}, {int(node4)})"
            print(f"{hyperedge_str:<60} {result['theoretical_prob']:.6f}   {result['empirical_prob']:.6f}")


# Main program
if __name__ == "__main__":
    # Set file path
    file_path = "tij_SFHH.dat_.csv"

    print("=== Hyperedge Steady-State Probability Convergence Validator (m=3) ===")
    print(f"Loading dataset: {file_path}")

    # Create validator instance
    validator = HyperedgeConvergenceValidatorM3(file_path)

    # Step 1: Load and prepare data
    validator.load_and_prepare_data()

    # Step 2: Calculate node properties
    validator.calculate_activity()
    validator.calculate_vulnerability()

    # Step 3: Validate convergence
    validator.validate_convergence(T_max=2 * 10 ** 9)

    print("\n=== Analysis Complete ===")