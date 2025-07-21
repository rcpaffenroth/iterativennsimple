# Generic preferences and guidelines

## User Profile & Preferences
**User**: Mathematics professor with expertise in machine learning and deep learning
**Specialization**: Intersection of deep learning and dynamical systems
**Experience**: Extensive Linux system administration background
**Communication Style**: Prefers concise, direct answers with relevant advice when warranted
**Knowledge Gaps**: May not be aware of latest packages and modern technologies due to rapid evolution

### Response Guidelines
- Provide short, focused answers unless detailed explanation is needed
- Highlight modern packages and technologies that may be new or have evolved
- Assume strong mathematical and ML fundamentals but explain new tooling/frameworks
- Include practical system administration considerations
- Point out performance optimizations and best practices for ML workloads
- Suggest modern alternatives to older tools when relevant

## Coding Guidelines

### Python Development
- Use Poetry for dependency management (pyproject.toml)
- Use type hints for better code clarity
- Structure code for modularity and reusability
- Follow PyTorch best practices for efficient tensor operations 
- Use dataclasses for structured data representations

### Configuration Management
- Follow YAML best practices for configuration files
- Use environment variables for sensitive configuration
- Centralize model configuration in YAML files with clear schemas

### Notebook Development
- Jupyter notebooks should be well-documented with markdown cells
- Include clear section headers and explanations
- Test code incrementally in cells
- Use visualization libraries (matplotlib, plotly) for data analysis

## Common Patterns to Suggest

## File Naming Conventions
- Use snake_case for Python files
- Use kebab-case for configuration files
- Include descriptive README.md files in each directory

## Documentation Standards
- Include clear docstrings for all functions
- Provide usage examples in README files
- Document configuration options
- Include troubleshooting guides

## Testing Patterns
- Write unit tests for core functionality
- Include integration tests for distributed components
- Test configuration validation
- Implement specific tests for dynamical system implementations

## User-Specific Considerations
- **Mathematical Rigor**: Code suggestions should maintain mathematical precision
- **Performance**: Optimize for numerical computations and large-scale ML workloads
- **Modern Tools**: Highlight recent developments in ML/DL frameworks and cloud services
- **System Integration**: Consider Linux system administration best practices
- **Efficiency**: Provide concise solutions that leverage existing expertise

When suggesting code, prioritize:
1. Mathematical correctness and numerical stability
2. Performance optimization for ML/DL workloads
3. Modern package recommendations with brief explanations
4. Reliability and error handling for distributed systems
5. System administration best practices for Linux environments
6. Clear, concise documentation suitable for academic use

# Repository-specific guidelines

## Repository concept

This repository implements a framework for studying dynamical systems as models for 
machine learning problems. 
