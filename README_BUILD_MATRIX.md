# FBGEMM CI Build Matrix Generator

This script generates build matrices for FBGEMM continuous integration workflows. It provides a configurable way to generate matrices for different build variants, targets, and modes.

## Overview

The `BuildMatrixGenerator` class takes three main parameters:
- **Build Variant**: The type of build (cpu, cuda, rocm, genai, hstu)
- **Build Target**: The specific target configuration (default, genai, hstu)  
- **Mode**: The type of operation (build, test, bench)

## Usage

### Command Line Interface

```bash
python3 generate_build_matrix.py --variant VARIANT --target TARGET --mode MODE [--output FILE]
```

#### Parameters

- `--variant`: Build variant
  - `cpu`: CPU-only builds
  - `cuda`: CUDA GPU builds
  - `rocm`: ROCm GPU builds  
  - `genai`: GenAI-specific builds
  - `hstu`: HSTU-specific builds

- `--target`: Build target
  - `default`: Standard build configuration
  - `genai`: GenAI-optimized configuration
  - `hstu`: HSTU-optimized configuration

- `--mode`: Build mode
  - `build`: Build-only matrix
  - `test`: Testing matrix (includes Python versions)
  - `bench`: Benchmarking matrix (reduced matrix for performance)

- `--output`: Optional output file (default: stdout)

#### Examples

```bash
# Generate CPU build matrix
python3 generate_build_matrix.py --variant cpu --target default --mode build

# Generate CUDA test matrix
python3 generate_build_matrix.py --variant cuda --target default --mode test

# Generate GenAI benchmark matrix
python3 generate_build_matrix.py --variant genai --target genai --mode bench

# Save to file
python3 generate_build_matrix.py --variant rocm --target default --mode test --output rocm_test_matrix.json
```

### Programmatic Usage

```python
from generate_build_matrix import BuildMatrixGenerator, BuildVariant, BuildTarget, Mode

# Using enums
generator = BuildMatrixGenerator(BuildVariant.CUDA, BuildTarget.DEFAULT, Mode.TEST)
matrix_json = generator.generate()
matrix_dict = generator.generate_dict()

# Using strings
generator = BuildMatrixGenerator("cuda", "default", "test")
matrix_json = generator.generate()
```

## Configuration Details

### Host Machines

The script includes predefined host machine configurations for each build variant:

- **CPU**: x86 and ARM instances
- **CUDA**: Various GPU instances with different CUDA versions (12.6.3, 12.8.1, 12.9.1)
- **ROCm**: ROCm-enabled instances with different ROCm versions (6.1, 6.2)
- **GenAI**: Memory-optimized instances for GenAI workloads
- **HSTU**: High-memory instances for HSTU workloads

### Compilers

- **CPU**: GCC (9.5.0, 14.1.0) and Clang (16.0.6)
- **GPU variants**: GCC and Clang (simplified configuration)

### Python Versions

Supports Python 3.9, 3.10, 3.11, 3.12, and 3.13 for test and benchmark modes.

### Build Matrix Structure

The generated JSON includes:

```json
{
  "matrix": {
    "host-machine": [...],
    "compiler": [...],
    "python-version": [...],  // Only for test/bench modes
    "library-type": [...]     // Only for CPU builds
  },
  "exclude": [...],           // Exclusion rules
  "_metadata": {              // Generation metadata
    "build_variant": "...",
    "build_target": "...",
    "mode": "..."
  }
}
```

## Validation Rules

The script includes validation to ensure compatible configurations:

- GenAI build target requires GenAI, CUDA, or ROCm variant
- HSTU build target requires HSTU or CUDA variant
- Invalid combinations will raise a `ValueError`

## Target Filtering

The script automatically filters host machines based on the build target:

- `default` target: Uses machines with no specific target or "default" target
- `genai` target: Uses only GenAI-optimized machines
- `hstu` target: Uses only HSTU-optimized machines

## Mode-Specific Behavior

- **Build mode**: Basic matrix without Python versions
- **Test mode**: Includes all Python versions for comprehensive testing
- **Bench mode**: Reduced matrix (single Python version) for performance testing

## Testing

Run the test script to see examples:

```bash
python3 test_build_matrix.py
```

This will demonstrate various matrix generations and validation scenarios.

## Integration with CI

This script can be integrated into GitHub Actions workflows:

```yaml
- name: Generate Build Matrix
  id: matrix
  run: |
    MATRIX=$(python3 scripts/generate_build_matrix.py --variant cuda --target default --mode test)
    echo "matrix=$MATRIX" >> $GITHUB_OUTPUT

- name: Build
  strategy:
    matrix: ${{ fromJson(steps.matrix.outputs.matrix) }}
  # ... rest of job configuration
```