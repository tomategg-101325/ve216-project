# ve216-project

## Introduction

A **signal separation neural network** to separate two sinusoidal signals with distinct frequencies from their linear combination.

Part of the course project for *Ve216 Introduction to Signals and Systems* in Spring 2026 at SJTU GC.

## How to Run

It is recommended to run the project on a **Linux** system.

1. Enter a Python virtual environment at your preference. Run `pip install -r requirements.txt` to install the necessary libraries.

2. Then, at the root directory, run

- `make train` to generate training data, train the model, and export the model weights to the file `separator_model_weights.pth`;
- `make run` to simulate with an arbitrary signal and plot the results accordingly;
- `make test` to train and simulate.

