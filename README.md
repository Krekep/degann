# DEGANN

[![Check tests](https://github.com/Krekep/degann/actions/workflows/tests.yml/badge.svg)](https://github.com/Krekep/degann/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/Krekep/degann/blob/main/LICENSE)
[![Package](https://img.shields.io/badge/pypi%20package-1.1-%233776ab)](https://pypi.org/project/degann/)

**DEGANN** is a library generating neural networks for approximating solutions to differential equations. As a backend for working with neural networks, tensorflow is used, but with the ability to expand with your own tools.

**Features**
- Generation of neural networks by parameters.
- Construction of tables with the numerical solution of ordinary differential equations of the first order
- Construction of tables with numerical solution of systems of ordinary differential equations of the first order
- Choosing the Best Neural Network from Several for Fixed Training Parameters
- Iterating over training parameters with choosing the best neural network for each set
- Export Neural Networks as a function in C++
- Export Neural Networks as a Parameter Set
- Import Neural Networks from a Parameter Set
- Building a dataset with complete training results for approximating the solution of a differential equation for each neural network that participated in training
- Advanced search for optimal topology using a language that describes the topology of a neural network
- Random search for optimal neural network topology
- Method for simulating annealing of optimal neural network topology
- Expert system for automatic selection of optimal parameters for algorithms for searching neural network topologies

## Install

### Manual
Download the repository as a zip archive, unpack and run the command from the root of the repository
```bash
pip install -r requirements.txt
```
This will download and install all the dependencies for the project, then you can use the source code of the library (e.g. create project nearby to the `degann` folder)

### Via pip
Use command
```bash
pip install degann
```
Now you can use the `degann` package
