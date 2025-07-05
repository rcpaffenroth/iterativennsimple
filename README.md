# Iterative Neural Networks (Simple)

A streamlined implementation exploring the connection between neural networks and dynamical systems through weight sharing and iterative architectures.

## Overview

This repository demonstrates how neural networks with shared weights across layers behave like discrete dynamical systems. By constraining traditional multilayer perceptrons (MLPs) to use the same weight matrix at each layer, we create iterative neural networks that exhibit rich mathematical behavior.

## Prerequisites

This README assumes basic familiarity with:
- GitHub, VS Code, and Codespaces
- Python programming and Jupyter notebooks
- Linear algebra and basic neural network concepts

**Helpful Resources**:
- [Codespaces tutorial](https://www.youtube.com/watch?v=ozuDPmcC1io&list=PLmsFUfdnGr3wTl-NCblzcrEv2lFSX975-&index=1)
- [VS Code documentation](https://code.visualstudio.com/docs)
- [Codespaces documentation](https://docs.github.com/en/codespaces/guides)

## Mathematical Background

### Traditional MLPs vs. Iterative Networks

**Standard MLP**:
```
σ(W_L σ(W_{L-1} ... σ(W_1 x + b_1) ... + b_{L-1}) + b_L)
```

**Iterative Network (Weight Sharing)**:
```
σ(W σ(W ... σ(W x + b) ... + b) + b)
```

The key insight: by sharing weights W and bias b across all layers, we create a discrete dynamical system that iterates the function f(x) = σ(Wx + b).

## Repository Structure

### Core Implementation (`iterativennsimple/`)

- **`Sequential2D.py`**: Main iterative network implementation supporting 2D problems
- **`Sequential1D.py`**: Simplified 1D version for basic experimentation  
- **`MaskedLinear.py`**: Linear layers with learnable masking for sparse connectivity
- **`SparseLinear.py`**: Efficient sparse linear transformations
- **`bmatrix.py`**: Utilities for block matrix operations

### Notebooks (`notebooks/`)

#### Core Tutorial Sequence
1. **`1-rcp-visualize-data.ipynb`**: Data visualization fundamentals
2. **`2-rcp-seqential-2D-problems.ipynb`**: Introduction to 2D dynamical systems
3. **`3-rcp-iterated-2D-problems.ipynb`**: Iterative approaches to 2D problems
4. **`4-rcp-MLP.ipynb`**: Comparison between standard MLPs and iterative networks
5. **`5-rcp-pulled-apart.ipynb`**: Detailed analysis of network components

#### Advanced Applications (`notebooks/advanced/`)
- **`2b-rcp-iterated-model-train-MNIST.ipynb`**: MNIST classification with iterative networks
- **`3-rcp-iterated-model-fish.ipynb`**: Complex pattern recognition tasks
- **`6-rcp-computational-performance.ipynb`**: Performance analysis and optimization

#### Presentations (`notebooks/presentations/`)
- **`dynamical_systems.ipynb`**: Interactive exploration of dynamical systems theory, including:
  - Bifurcation diagrams
  - Mandelbrot set generation via neural networks
  - Connections between chaos theory and deep learning

### Utility Scripts (`scripts/`)
- **`post-create.sh`**: Environment setup script
- **`install-torch-sparse.sh`**: Sparse tensor library installation
- **`wandb/`**: Weights & Biases integration for experiment tracking

### Testing (`tests/`)
Comprehensive test suite ensuring reproducibility of all notebook results.

## Key Concepts Explored

### 1. Weight Sharing as Dynamical Systems
Learn how constraining neural networks to share weights creates discrete dynamical systems with rich mathematical properties.

### 2. Mathematical Connections
Explore surprising connections between:
- Neural network training and fixed-point theory
- Chaos theory and deep learning
- Fractal geometry and network behavior

### 3. Practical Applications
Apply iterative networks to real problems:
- Image classification (MNIST)
- Pattern recognition
- Time series analysis

## Quick Start

### 1. Environment Setup
Open a terminal and run the setup script:
```bash
scripts/post-create.sh
```

### 2. Activate Python Environment
```bash
source venv/bin/activate
```

### 3. Start with Core Notebooks
Begin with the numbered sequence in `notebooks/`:
1. **Data Visualization**: `1-rcp-visualize-data.ipynb`
2. **2D Systems**: `2-rcp-seqential-2D-problems.ipynb` 
3. **Iterative Approaches**: `3-rcp-iterated-2D-problems.ipynb`

### 4. Explore Advanced Topics
- Try the **Dynamical Systems** presentation for interactive mathematical exploration
- Experiment with **MNIST classification** using iterative networks
- Analyze **computational performance** characteristics

## Development Philosophy

This repository prioritizes:
- **Mathematical Rigor**: Precise connections between theory and implementation
- **Educational Clarity**: Clear progression from basic concepts to advanced applications
- **Reproducibility**: All results are tested and verified
- **Modern Tools**: Uses contemporary Python ML stack (PyTorch, Plotly, Poetry)

## Modern Dependencies

The project uses cutting-edge tools:
- **PyTorch 2.5+**: Latest deep learning framework
- **Plotly 5.24+**: Interactive visualizations
- **Poetry**: Modern Python dependency management
- **Weights & Biases**: Experiment tracking and visualization
- **pytest + nbmake**: Automated notebook testing

## Next Steps

This repository serves as a foundation for deeper exploration. Once comfortable with the concepts:

1. **Local Development**: Set up the environment on your own machine
2. **Extended Research**: Explore connections to your specific domain
3. **Advanced Topics**: Investigate sparse networks, optimization theory, or chaos applications
4. **Collaboration**: Contact [rcpaffenroth@wpi.edu](mailto:rcpaffenroth@wpi.edu) for research opportunities

## Research Context

This work sits at the intersection of:
- **Dynamical Systems Theory**: Fixed points, stability, chaos
- **Deep Learning**: Architecture design, optimization
- **Applied Mathematics**: Numerical analysis, linear algebra

The goal is to bridge pure mathematical theory with practical machine learning applications, providing insights valuable for both theoretical understanding and engineering practice.


