# DEGANN

[![Check tests](https://github.com/Krekep/degann/actions/workflows/tests.yml/badge.svg)](https://github.com/Krekep/degann/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/Krekep/degann/blob/main/LICENSE)

**DEGANN** is a library generating neural networks for approximating solutions to differential equations. As a backend for working with neural networks, tensorflow is used, but with the ability to expand with your own tools.

**Features**
- Generation of neural networks by parameters.
- Construction of tables with the numerical solution of ordinary differential equations of the first order
- Construction of tables with numerical solution of systems of ordinary differential equations of the first order
- Choosing the Best Neural Network from Several for Fixed Training Parameters
- Iterating over training parameters with choosing the best neural network for each set
- Export neural networks as a function in c++
- Export Neural Networks as a Parameter Set
- Import Neural Networks from a Parameter Set
- Building a dataset with complete training results for approximating the solution of a differential equation for each neural network that participated in training
