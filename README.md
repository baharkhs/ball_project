# Minimal Molecular Dynamics Simulation

This package provides a simple, modular, and configuration-driven molecular dynamics simulation code written in Python. It demonstrates key concepts such as particle interactions via the Lennard-Jones potential, periodic boundary conditions, and visualization of molecular trajectories and system temperature.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Examples](#examples)
- [Testing](#testing)
- [Future Work](#future-work)
- [License](#license)

## Introduction

This project is designed to be both educational and modular. It simulates a simple molecular dynamics (MD) system where particles (atoms) interact through simplified potentials (e.g., Lennard-Jones) within a cylindrical well. All simulation parameters—such as the time step, well dimensions, molecule definitions, interaction parameters, and default particle properties—are specified in external JSON configuration files. This means you can simulate a single particle, a water molecule (composed of oxygen and hydrogen atoms), or multiple molecules by simply changing the input file without modifying the code.

## Features

- **Modular Design:**  
  Separate modules for particles (`Ball`), simulation boundaries (`Well`), simulation dynamics (`Simulation`), and visualization (`SimulationPlotter`).

- **Configuration-Driven:**  
  All parameters are set in JSON files. The same code runs for different scenarios (single particle, one water molecule, or multiple molecules) by changing the configuration.

- **Visualization:**  
  Matplotlib is used to animate the simulation in 3D (showing particle positions and bonds) along with a 2D plot for the system temperature over time.

- **Interparticle Interactions:**  
  Implements Lennard-Jones forces (and optionally Coulombic interactions) along with wall repulsion forces, while applying periodic boundary conditions.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/ball_project.git
   cd ball_project
   
2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Linux/MacOS
    .venv\Scripts\activate      # On Windows
3. **Install required dependencies:**
    ```bash
   pip install -r requirements.txt
   pip install .
   
## Usage

To run the simulation using the default configuration (for example, two water molecules), simply run:

    ```bash
    python main.py

This command:
- Loads `config.json`
- Creates a Simulation object using the configuration
- Launches the visualization with the SimulationPlotter


## Examples

The `examples/` folder contains alternative configurations and scripts:

- **Single Particle Example:**  
  Run the script:
  ```bash
  python examples/run_single_ball.py
  
This uses config_single_ball.json to simulate a single particle.

- **One Water Molecule Example**  
  Run the script:
  ```bash
  python examples/run_one_water.py
  
 This uses config_one_water.json to simulate a single water molecule (one oxygen with two hydrogens).

Note: The core code remains unchanged. Only the JSON input changes, allowing you to simulate different scenarios without modifying the code.

## Testing
Unit tests are provided in the `tests/` folder.

 
These tests cover:
-Initialization of Balls and Wells
-Force calculations and periodic boundary conditions 
-Simulation dynamics and integration
-Visualization data (such as paths and temperature)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
