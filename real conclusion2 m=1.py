import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import seaborn as sns
from scipy.stats import linregress
import csv


class SteadyStateProbabilityValidator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.nodes = set()
        self.time_windows = None
        self.a_i = {}  # Node activity for m=1
        self.b_i = {}  # Node vulnerability for m=1
        self.unique_edges = set()  # Store all unique edges in the dataset
        self.edge_theoretical_probs = {}  # Theoretical steady-state probabilities for each edge
        self.edge_empirical_probs = {}  # Empirical steady-state probabilities for each edge
        self.n = 0  # Number of nodes
        self.m = 1  # Order of interaction (m=1 for pairwise)

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

        # Extract all unique edges
        for _, row in self.df.iterrows():
            edge = tuple(sorted([row['node1'], row['node2']]))
            self.unique_edges.add(edge)
        print(f"Identified {len(self.unique_edges)} unique edges")

    def calculate_activity(self):
        """Calculate node activity for m=1 (a_i)"""
        # Track which time windows each node participates in an interaction
        node_active_windows = defaultdict(set)

        for _, row in self.df.iterrows():
            node_active_windows[row['node1']].add(row['time_window'])
            node_active_windows[row['node2']].add(row['time_window'])

        # Calculate activity = active windows / total windows
        total_windows = len(self.time_windows)
        for node in self.nodes:
            active_count = len(node_active_windows.get(node, set()))
            self.a_i[node] = active_count / total_windows

        print("Node activity (m=1) calculation completed")

    def calculate_vulnerability(self, inactive_threshold=1000):
        """Calculate node vulnerability for m=1 (b_i)"""
        # Track interruption count for each node
        interruption_count = defaultdict(int)

        # Find all time windows where each node is active
        node_active_windows = defaultdict(list)
        for _, row in self.df.iterrows():
            node_active_windows[row['node1']].append(row['time_window'])
            node_active_windows[row['node2']].append(row['time_window'])

        # For each node, check for interruptions
        for node in tqdm(self.nodes, desc="Calculating vulnerability"):
            active_windows = sorted(node_active_windows[node])
            for i in range(len(active_windows)):
                current_window = active_windows[i]

                # Only consider events that have at least 1000 future windows
                if current_window <= max(self.time_windows) - inactive_threshold:
                    # Check if there's another interaction within the next 1000 windows
                    found_next = False
                    for j in range(i + 1, len(active_windows)):
                        if active_windows[j] <= current_window + inactive_threshold:
                            found_next = True
                            break

                    # If no interaction found within next 1000 windows, count as interruption
                    if not found_next:
                        interruption_count[node] += 1

        # Calculate vulnerability
        valid_windows = len(self.time_windows) - inactive_threshold
        for node in self.nodes:
            if valid_windows > 0:
                self.b_i[node] = interruption_count[node] / valid_windows
            else:
                self.b_i[node] = 0

        print("Node vulnerability (m=1) calculation completed")

    def calculate_theoretical_probabilities(self):
        """Calculate theoretical steady-state probabilities for each edge"""
        for edge in self.unique_edges:
            node1, node2 = edge
            numerator = self.a_i[node1] + self.a_i[node2]
            denominator = numerator + self.b_i[node1] + self.b_i[node2]
            self.edge_theoretical_probs[edge] = numerator / denominator if denominator > 0 else 0

        print("Theoretical steady-state probabilities calculated")

    def simulate_single_edge_dynamics(self, T, a1, a2, b1, b2):
        """Simulate the dynamics of a single edge using event-driven approach"""
        # Calculate rates
        λ = (a1 + a2) / self.n  # Activation rate
        μ = (b1 + b2) / self.n  # Deactivation rate

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
        """Validate convergence for multiple edges in a single figure"""
        # Calculate theoretical probabilities
        self.calculate_theoretical_probabilities()

        # Select top 4 most frequent edges
        edge_counts = defaultdict(int)
        for _, row in self.df.iterrows():
            edge = tuple(sorted([row['node1'], row['node2']]))
            edge_counts[edge] += 1

        # Sort edges by frequency
        sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
        top_edges = [edge for edge, count in sorted_edges[:4]]

        print(f"\nSelected top {len(top_edges)} edges for validation:")
        for i, edge in enumerate(top_edges):
            node1, node2 = edge
            print(f"{i + 1}. Edge (node {node1}, node {node2}): {edge_counts[edge]} interactions")

        # Create figure with 4 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Steady-State Probability Convergence for Selected Hyperedges (T=10000>>403^1)', fontsize=16)

        # Validate convergence for each edge
        results = []
        for i, edge in enumerate(top_edges):
            node1, node2 = edge
            a1, a2 = self.a_i[node1], self.a_i[node2]
            b1, b2 = self.b_i[node1], self.b_i[node2]

            # Print node properties
            print(f"\nEdge (node {node1}, node {node2}) properties:")
            print(f"  Node {node1}: a_i = {a1:.6f}, b_i = {b1:.6f}")
            print(f"  Node {node2}: a_i = {a2:.6f}, b_i = {b2:.6f}")
            print(f"  Sum a = {a1 + a2:.6f}, Sum b = {b1 + b2:.6f}")

            # Calculate theoretical steady-state probability
            numerator = a1 + a2
            denominator = numerator + b1 + b2
            theoretical_prob = numerator / denominator if denominator > 0 else 0
            print(f"  Theoretical P∞ = {theoretical_prob:.6f}")

            # Set time points
            T_max = 10000
            T_points = np.logspace(0, np.log10(T_max), 50, dtype=int)
            T_points = np.unique(T_points)  # Ensure unique values

            # Store results
            empirical_probs = []
            exact_theoretical = []

            # Simulate for each time point
            for T in tqdm(T_points, desc=f"Simulating edge (node {node1}, node {node2})"):
                active_count = 0
                for _ in range(100):  # Reduced simulations for efficiency
                    if self.simulate_single_edge_dynamics(T, a1, a2, b1, b2):
                        active_count += 1
                empirical_probs.append(active_count / 100)

                # Calculate exact theoretical probability at T
                factor = 1 / self.n  # m! / n^m for m=1
                total_rate = factor * (a1 + a2 + b1 + b2)
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

            # Set title and labels - FIXED: Ensure integer formatting
            ax.set_title(f'Edge (node {int(node1)}, node {int(node2)})', fontsize=12)
            ax.set_xlabel('Time (T)')
            ax.set_ylabel('Edge Probability')
            ax.grid(True, which="both", linestyle='--', alpha=0.7)
            ax.legend()

            # Add annotation
            ax.annotate(f'P∞ = {theoretical_prob:.4f}',
                        xy=(0.95, 0.05), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

            # Store results for summary
            results.append({
                'edge': edge,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_probs[-1]  # Last point at T=10000
            })

        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig('edge_convergence_summary.png', dpi=300)
        print("\nVisualization saved to 'edge_convergence_summary.png'")

        # Print summary
        print("\n=== Validation Summary ===")
        print(f"{'Edge':<25} {'Theoretical':<12} {'Empirical':<12}")
        print("-" * 50)
        for result in results:
            node1, node2 = result['edge']
            edge_str = f"(node {int(node1)}, node {int(node2)})"
            print(f"{edge_str:<25} {result['theoretical_prob']:.6f}   {result['empirical_prob']:.6f}")


# Main program
if __name__ == "__main__":
    # Set file path
    file_path = "tij_SFHH.dat_.csv"

    print("=== Steady-State Probability Convergence Validator (m=1) ===")
    print(f"Loading dataset: {file_path}")

    # Create validator instance
    validator = SteadyStateProbabilityValidator(file_path)

    # Step 1: Load and prepare data
    validator.load_and_prepare_data()

    # Step 2: Calculate node properties
    validator.calculate_activity()
    validator.calculate_vulnerability()

    # Step 3: Validate convergence
    validator.validate_convergence()

    print("\n=== Analysis Complete ===")