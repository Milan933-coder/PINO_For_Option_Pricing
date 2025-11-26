Adaptive PINO Heston Option Pricing Engine (Simplified)

🎯 Project Summary

This project develops a state-of-the-art solution for pricing European options under the Heston Stochastic Volatility Model. We utilize a hybrid Physics-Informed Neural Operator (PINO), combining the high-speed processing of the Fourier Neural Operator (FNO) with the stability of Adaptive Learning (WamOL).

Core Goal: To create an instantaneous and accurate pricing model that can replace slow and unstable numerical methods (Monte Carlo and FFT) for real-time financial calibration.

The Problem

Traditional Heston pricing methods are computationally expensive. Monte Carlo is too slow, and Fast Fourier Transform (FFT) suffers from numerical instability and requires manual tuning for different volatility regimes.

Our Solution

We train the neural network to solve the Feynman-Kac PDE (the governing physics equation) across the entire parameter space. Advanced techniques used include:

FNO Backbone: Ensures efficient learning and high-speed prediction.

Hard Ansatz Transform: Guarantees structural stability by satisfying the terminal payoff boundary condition analytically.

WamOL Adaptive Loss: Automatically balances multiple physics constraints (PDE, Delta, Gamma smoothness) during training, optimizing for uniform accuracy.

📈 Current Results & Next Steps

Our initial tests confirm the framework's operational efficiency but highlight a capacity issue:

Metric

Finding

Status

Computational Speed

273.8X Speedup over Monte Carlo

CONFIRMED VIABLE

Accuracy (MAE)

$2.75 Mean Absolute Error

INSUFFICIENT

Next Step: The current model is limited by VRAM, preventing training on a necessary high-resolution grid (targeting $96 \times 48 \times 48$ resolution). The next phase requires high-VRAM computing resources to scale the network capacity and achieve financial-grade accuracy (target MAE $<\$0.10$).
