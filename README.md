# Attacks-Delay

A Python simulation project that models and analyzes packet delay in network systems under attack conditions.

## Overview

This project implements both theoretical and simulation models to study the impact of attacks on packet delay in queueing systems. It compares theoretical calculations with discrete-event simulation results to validate the mathematical model.

## Features

- **Theoretical Model**: Mathematical calculation of average packet delay under attack conditions
- **Discrete-Event Simulation**: SimPy-based simulation with realistic packet processing
- **Attack Modeling**: Configurable attack effectiveness and frequency
- **Timeout Handling**: Proper retransmission logic based on waiting time thresholds
- **Visualization**: Comparative plots of theoretical vs. simulated results

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SimPy

## Setup

1. Clone this repository:

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib simpy
   ```

## Parameters

Key simulation parameters (configurable in `test.py`):

- `mu = 10.0`: Service rate (packets/second)
- `lambda_a = 1.5`: Attack arrival rate
- `T = 0.5`: Timeout threshold for retransmission
- `normal_traffic_rates`: Range of normal traffic rates to test
- `attack_effectiveness_values`: Attack success probability values

## Model Description

The system models:
1. **Normal Traffic**: Packets arrive according to a Poisson process
2. **Attacks**: Corrupt packets in the queue with configurable effectiveness
3. **Service**: Single-server queue with exponential service times
4. **Retransmission**: Packets are retransmitted if corrupted or waiting time exceeds threshold

## Output

The simulation produces:
- Console output showing simulation progress
- A PNG plot comparing theoretical and simulated delays
- Statistical analysis of packet delays under different conditions
