#!/usr/bin/env python3

"""
Analyze the results of a simulation from the output JSON file.

This script reads the simulation output data, extracts trajectories or other
relevant metrics, and performs analysis or visualization (e.g., plotting ball paths).

Example usage:
    python analyze_output.py simulation_output.json
"""

import sys
import os
import json
import matplotlib.pyplot as plt


def load_output(output_path):
    """
    Load simulation output from a JSON file.

    Args:
        output_path (str): Path to the simulation output file.

    Returns:
        dict: Parsed simulation data as a dictionary.

    Raises:
        FileNotFoundError: If the output file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file not found: {output_path}")

    with open(output_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON from file {output_path}: {e}")
    return data


def plot_trajectories(output_data):
    """
    Plot the trajectories of balls based on simulation output.

    Args:
        output_data (dict): Parsed simulation output data.

    Raises:
        ValueError: If the output data format is invalid or missing required keys.
    """
    # Prepare the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Ball Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Extract ball paths and plot them
    has_data = False  # Track if any data is plotted
    for ball_id, ball_data in enumerate(output_data.get("balls", [])):
        if "path_segments" not in ball_data:
            raise ValueError(f"Ball {ball_id} data missing 'path_segments' key.")
        for segment in ball_data["path_segments"]:
            ax.plot(segment["x"], segment["y"], segment["z"], label=f"Ball {ball_id}")
            has_data = True

    if not has_data:
        print("Warning: No ball trajectories found to plot.")
    else:
        ax.legend()

    plt.show()


def main(output_path=None):
    """
    Main function to analyze simulation output.

    Args:
        output_path (str, optional): Path to the simulation output file. If None, defaults
        to 'examples/simulation_output.json'.

    Raises:
        Exception: For any errors during analysis or visualization.
    """
    # Resolve the output file path
    if output_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_path = os.path.join(project_root, "examples", "simulation_output.json")

    # Debugging: Print the resolved output file path
    print("Using output file:", output_path)

    try:
        # Load the simulation output
        output_data = load_output(output_path)

        # Perform analysis or visualization
        print("Analyzing simulation results...")
        plot_trajectories(output_data)
        print("Analysis complete.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")


if __name__ == "__main__":
    # Allow passing a custom output file via command line
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(output_file)
