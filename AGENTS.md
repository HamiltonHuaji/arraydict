# Implement arraydict library

Implement a lightweight, JAX-backed container called ArrayDict that stores a mapping of arrays or lists sharing leading batch dimensions. The required features are described in README.md.

The project should use hatchling.build for packaging and include basic type annotations. There should be a src/ directory for the source code and a tests/ directory for unit tests.

For testing, use an optional dependency group `dev` that includes torch and tensordict. The tests should cover all functionalities mentioned in README.md. For nested keys and values, use randomized data to ensure robustness. The test cases should have complex nested structures and varying depths.
