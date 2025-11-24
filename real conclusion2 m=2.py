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


class SteadyStateProbabilityValidatorM2:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.nodes = set()
        self.time_windows = None
        self.hyper_edges = defaultdict(list)  # Store hyperedges for each time window
        self.a_i = {}  # Node activity for m=2
        self.b_i = {}  # Node vulnerability for m=2
        self.unique_hyperedges = set()  # Store all unique hyperedges in the dataset
        self.hyperedge_theoretical_probs = {}  # Theoretical steady-state probabilities for each hyperedge
        self.n = 0  # Number of nodes
        self.m = 2  # Order of interaction (m=2 for three-node interactions)

    def load_and_prepare_data(self):
        """Load dataset and prepare analysis environment"""
        # Load CSV file
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset contains {len(self.df)} rows")

        # Convert node IDs to integers to avoid float representation
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
        print(f"Total time windows: {len(self.time_windows)} (from {min_win} to {max_win})")

        # Identify triangle interactions (m=2 hyperedges)
        self.identify_triangles()

        # Add empty time windows
        for time_win in self.time_windows:
            if time_win not in self.hyper_edges:
                self.hyper_edges[time_win] = []

        total_hyperedges = sum(len(v) for v in self.hyper_edges.values())
        print(f"Identified {total_hyperedges} triangle interactions")

    def identify_triangles(self):
        """Identify triangle interactions (m=2 hyperedges) in each time window"""
        print("Identifying triangle interactions...")

        # Group by time window
        grouped = self.df.groupby('time_window')

        # For each time window, identify triangles
        for time_win, group in tqdm(grouped, total=len(grouped)):
            # Create graph for this time window
            G = nx.Graph()
            for _, row in group.iterrows():
                G.add_edge(row['node1'], row['node2'])

            # Identify triangles (3 nodes with all pairwise connections)
            triangles = set()
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if G.has_edge(neighbors[i], neighbors[j]):
                            # Create sorted tuple for triangle
                            triangle = tuple(sorted([node, neighbors[i], neighbors[j]]))
                            triangles.add(triangle)

            self.hyper_edges[time_win] = list(triangles)

            # Add to unique hyperedges
            for triangle in triangles:
                self.unique_hyperedges.add(triangle)

    def calculate_activity(self):
        """Calculate node activity for m=2 (a_i)"""
        # Track which time windows each node participates in a triangle
        node_active_windows = defaultdict(set)

        for time_win, triangles in self.hyper_edges.items():
            for triangle in triangles:
                for node in triangle:
                    node_active_windows[node].add(time_win)

        # Calculate activity = active windows / total windows
        total_windows = len(self.time_windows)
        for node in self.nodes:
            active_count = len(node_active_windows.get(node, set()))
            self.a_i[node] = active_count / total_windows

        print("Node activity (m=2) calculation completed")

    def calculate_vulnerability(self, inactive_threshold=1000):
        """Calculate node vulnerability for m=2 (b_i)"""
        # Track interruption count for each node
        interruption_count = defaultdict(int)

        # Find all time windows where each node is active in a triangle
        node_active_windows = defaultdict(list)
        for time_win, triangles in self.hyper_edges.items():
            for triangle in triangles:
                for node in triangle:
                    node_active_windows[node].append(time_win)

        # For each node, check for interruptions
        for node in tqdm(self.nodes, desc="Calculating vulnerability"):
            active_windows = sorted(node_active_windows[node])
            for i in range(len(active_windows)):
                current_window = active_windows[i]

                # Only consider events that have at least 1000 future windows
                if current_window <= max(self.time_windows) - inactive_threshold:
                    # Check if there's another triangle within the next 1000 windows
                    found_next = False
                    for j in range(i + 1, len(active_windows)):
                        if active_windows[j] <= current_window + inactive_threshold:
                            found_next = True
                            break

                    # If no triangle found within next 1000 windows, count as interruption
                    if not found_next:
                        interruption_count[node] += 1

        # Calculate vulnerability
        valid_windows = len(self.time_windows) - inactive_threshold
        for node in self.nodes:
            if valid_windows > 0:
                self.b_i[node] = interruption_count[node] / valid_windows
            else:
                self.b_i[node] = 0

        print("Node vulnerability (m=2) calculation completed")

    def calculate_theoretical_probabilities(self):
        """Calculate theoretical steady-state probabilities for each hyperedge"""
        for hyperedge in self.unique_hyperedges:
            # Calculate numerator: sum of a_i for nodes in hyperedge
            numerator = sum(self.a_i[node] for node in hyperedge)

            # Calculate denominator: sum of (a_i + b_i) for nodes in hyperedge
            denominator = sum(self.a_i[node] + self.b_i[node] for node in hyperedge)

            # Avoid division by zero
            if denominator > 0:
                self.hyperedge_theoretical_probs[hyperedge] = numerator / denominator
            else:
                self.hyperedge_theoretical_probs[hyperedge] = 0

        print("Theoretical steady-state probabilities calculated")

    def simulate_single_hyperedge_dynamics(self, T, a_values, b_values):
        """Simulate the dynamics of a single hyperedge using event-driven approach"""
        # Calculate rates
        λ = sum(a_values) / (self.n ** self.m)  # Activation rate
        μ = sum(b_values) / (self.n ** self.m)  # Deactivation rate

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

    def validate_convergence(self):
        """Validate convergence for multiple hyperedges in a single figure"""
        # Calculate theoretical probabilities
        self.calculate_theoretical_probabilities()

        # Select top 4 most frequent hyperedges
        hyperedge_counts = defaultdict(int)
        for time_win, triangles in self.hyper_edges.items():
            for triangle in triangles:
                hyperedge_counts[triangle] += 1

        # Sort hyperedges by frequency
        sorted_hyperedges = sorted(hyperedge_counts.items(), key=lambda x: x[1], reverse=True)
        top_hyperedges = [hyperedge for hyperedge, count in sorted_hyperedges[:4]]

        print(f"\nSelected top {len(top_hyperedges)} hyperedges for validation:")
        for i, hyperedge in enumerate(top_hyperedges):
            node1, node2, node3 = hyperedge
            print(
                f"{i + 1}. Hyperedge (node {int(node1)}, node {int(node2)}, node {int(node3)}): {hyperedge_counts[hyperedge]} interactions")

        # Create figure with 4 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('(b) Steady-State Probability Convergence for Selected Hyperedges (m=2, T=10000000>>403^2)', fontsize=16)

        # Validate convergence for each hyperedge
        results = []
        for i, hyperedge in enumerate(top_hyperedges):
            node1, node2, node3 = hyperedge
            a1, a2, a3 = self.a_i[node1], self.a_i[node2], self.a_i[node3]
            b1, b2, b3 = self.b_i[node1], self.b_i[node2], self.b_i[node3]

            # Print node properties
            print(f"\nHyperedge (node {int(node1)}, node {int(node2)}, node {int(node3)}) properties:")
            print(f"  Node {int(node1)}: a_i = {a1:.6f}, b_i = {b1:.6f}")
            print(f"  Node {int(node2)}: a_i = {a2:.6f}, b_i = {b2:.6f}")
            print(f"  Node {int(node3)}: a_i = {a3:.6f}, b_i = {b3:.6f}")
            print(f"  Sum a = {a1 + a2 + a3:.6f}, Sum b = {b1 + b2 + b3:.6f}")

            # Calculate theoretical steady-state probability
            numerator = a1 + a2 + a3
            denominator = numerator + b1 + b2 + b3
            theoretical_prob = numerator / denominator if denominator > 0 else 0
            print(f"  Theoretical P∞ = {theoretical_prob:.6f}")

            # Set time points
            T_max = 10000000  # T >> n^m ≈ 160,000
            T_points = np.logspace(0, np.log10(T_max), 50, dtype=int)
            T_points = np.unique(T_points)  # Ensure unique values

            # Store results
            empirical_probs = []
            exact_theoretical = []

            # Simulate for each time point
            for T in tqdm(T_points, desc=f"Simulating hyperedge (node {int(node1)}, node {int(node2)}, node {int(node3)})"):
                active_count = 0
                for _ in range(100):  # Reduced simulations for efficiency
                    if self.simulate_single_hyperedge_dynamics(T, [a1, a2, a3], [b1, b2, b3]):
                        active_count += 1
                empirical_probs.append(active_count / 100)

                # Calculate exact theoretical probability at T
                factor = math.factorial(1) / (self.n ** self.m)  # (m-1)! / n^m for m=2
                total_rate = factor * (a1 + a2 + a3 + b1 + b2 + b3)
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
            ax.set_title(f'Hyperedge (node {int(node1)}, node {int(node2)}, node {int(node3)})', fontsize=12)
            ax.set_xlabel('Time Step(T)')
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
                'empirical_prob': empirical_probs[-1]  # Last point at T=10000000
            })

        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig('hyperedge_convergence_summary_m2.png', dpi=300)
        print("\nVisualization saved to 'hyperedge_convergence_summary_m2.png'")

        # Print summary
        print("\n=== Validation Summary ===")
        print(f"{'Hyperedge':<40} {'Theoretical':<12} {'Empirical':<12}")
        print("-" * 65)
        for result in results:
            node1, node2, node3 = result['hyperedge']
            hyperedge_str = f"(node {int(node1)}, node {int(node2)}, node {int(node3)})"
            print(f"{hyperedge_str:<40} {result['theoretical_prob']:.6f}   {result['empirical_prob']:.6f}")


# Main program
if __name__ == "__main__":
    # Set file path
    file_path = "tij_SFHH.dat_.csv"

    print("=== Steady-State Probability Convergence Validator (m=2) ===")
    print(f"Loading dataset: {file_path}")

    # Create validator instance
    validator = SteadyStateProbabilityValidatorM2(file_path)

    # Step 1: Load and prepare data
    validator.load_and_prepare_data()

    # Step 2: Calculate node properties
    validator.calculate_activity()
    validator.calculate_vulnerability()

    # Step 3: Validate convergence
    validator.validate_convergence()

    print("\n=== Analysis Complete ===")