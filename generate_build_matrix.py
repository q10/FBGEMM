#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum


class BuildVariant(Enum):
    """Supported build variants for FBGEMM CI."""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    GENAI = "genai"
    HSTU = "hstu"


class BuildTarget(Enum):
    """Supported build targets for FBGEMM CI."""
    DEFAULT = "default"
    GENAI = "genai"
    HSTU = "hstu"


class Mode(Enum):
    """Supported build modes for FBGEMM CI."""
    BUILD = "build"
    TEST = "test"
    BENCH = "bench"


@dataclass
class HostMachine:
    """Configuration for a host machine."""
    arch: str
    instance: str
    build_target: Optional[str] = None
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None


@dataclass
class Compiler:
    """Configuration for a compiler."""
    name: str
    version: str


class BuildMatrixGenerator:
    """
    Generator for FBGEMM CI build matrices.
    
    This class generates build matrices for different build variants, targets, and modes
    used in FBGEMM continuous integration workflows.
    """
    
    # Common configurations
    PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]
    
    # Host machine configurations
    HOST_MACHINES = {
        BuildVariant.CPU: [
            HostMachine(arch="x86", instance="linux.12xlarge"),
            HostMachine(arch="arm", instance="linux.arm64.2xlarge"),
        ],
        BuildVariant.CUDA: [
            HostMachine(arch="x86", instance="linux.24xlarge", build_target="default", cuda_version="12.6.3"),
            HostMachine(arch="x86", instance="linux.24xlarge", build_target="default", cuda_version="12.8.1"),
            HostMachine(arch="x86", instance="linux.24xlarge", build_target="default", cuda_version="12.9.1"),
            HostMachine(arch="x86", instance="linux.12xlarge.memory", build_target="genai", cuda_version="12.6.3"),
            HostMachine(arch="x86", instance="linux.12xlarge.memory", build_target="genai", cuda_version="12.8.1"),
            HostMachine(arch="x86", instance="linux.12xlarge.memory", build_target="genai", cuda_version="12.9.1"),
            HostMachine(arch="x86", instance="linux.24xlarge.memory", build_target="hstu", cuda_version="12.9.1"),
        ],
        BuildVariant.ROCM: [
            HostMachine(arch="x86", instance="linux.rocm.gpu", build_target="default", rocm_version="6.1"),
            HostMachine(arch="x86", instance="linux.rocm.gpu", build_target="default", rocm_version="6.2"),
            HostMachine(arch="x86", instance="linux.rocm.gpu", build_target="genai", rocm_version="6.2"),
        ],
        BuildVariant.GENAI: [
            HostMachine(arch="x86", instance="linux.12xlarge.memory", build_target="genai", cuda_version="12.6.3"),
            HostMachine(arch="x86", instance="linux.12xlarge.memory", build_target="genai", cuda_version="12.8.1"),
            HostMachine(arch="x86", instance="linux.12xlarge.memory", build_target="genai", cuda_version="12.9.1"),
        ],
        BuildVariant.HSTU: [
            HostMachine(arch="x86", instance="linux.24xlarge.memory", build_target="hstu", cuda_version="12.9.1"),
        ],
    }
    
    # Compiler configurations
    COMPILERS = {
        BuildVariant.CPU: [
            Compiler(name="gcc", version="9.5.0"),
            Compiler(name="gcc", version="14.1.0"),
            Compiler(name="clang", version="16.0.6"),
        ],
        BuildVariant.CUDA: ["gcc", "clang"],
        BuildVariant.ROCM: ["clang"],
        BuildVariant.GENAI: ["gcc", "clang"],
        BuildVariant.HSTU: ["gcc", "clang"],
    }
    
    # Library types for CPU builds
    LIBRARY_TYPES = ["static", "shared"]
    
    def __init__(self, build_variant: Union[BuildVariant, str], 
                 build_target: Union[BuildTarget, str], 
                 mode: Union[Mode, str]):
        """
        Initialize the BuildMatrixGenerator.
        
        Args:
            build_variant: The build variant (cpu, cuda, rocm, genai, hstu)
            build_target: The build target (default, genai, hstu)
            mode: The build mode (build, test, bench)
        """
        self.build_variant = BuildVariant(build_variant) if isinstance(build_variant, str) else build_variant
        self.build_target = BuildTarget(build_target) if isinstance(build_target, str) else build_target
        self.mode = Mode(mode) if isinstance(mode, str) else mode
        
        # Validate compatibility
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate that the build configuration is compatible."""
        # GenAI build target requires GenAI or CUDA variant
        if self.build_target == BuildTarget.GENAI:
            if self.build_variant not in [BuildVariant.GENAI, BuildVariant.CUDA, BuildVariant.ROCM]:
                raise ValueError(f"GenAI build target requires GenAI, CUDA, or ROCm variant, got {self.build_variant}")
        
        # HSTU build target requires HSTU or CUDA variant
        if self.build_target == BuildTarget.HSTU:
            if self.build_variant not in [BuildVariant.HSTU, BuildVariant.CUDA]:
                raise ValueError(f"HSTU build target requires HSTU or CUDA variant, got {self.build_variant}")
    
    def _get_filtered_host_machines(self) -> List[HostMachine]:
        """Get host machines filtered by build target."""
        host_machines = self.HOST_MACHINES.get(self.build_variant, [])
        
        if self.build_target == BuildTarget.DEFAULT:
            return [hm for hm in host_machines if hm.build_target in [None, "default"]]
        else:
            return [hm for hm in host_machines if hm.build_target == self.build_target.value]
    
    def _get_compilers(self) -> List[Union[Compiler, str]]:
        """Get compilers for the build variant."""
        return self.COMPILERS.get(self.build_variant, [])
    
    def _generate_cpu_matrix(self) -> Dict[str, Any]:
        """Generate build matrix for CPU variant."""
        host_machines = self._get_filtered_host_machines()
        compilers = self._get_compilers()
        
        matrix = {
            "host-machine": [asdict(hm) for hm in host_machines],
            "library-type": self.LIBRARY_TYPES,
            "compiler": [asdict(c) if isinstance(c, Compiler) else c for c in compilers],
        }
        
        # Add exclusions for ARM + GCC 9.5.0 (ARM SVE support doesn't exist)
        excludes = [
            {
                "host-machine": {"arch": "arm", "instance": "linux.arm64.2xlarge"},
                "compiler": {"name": "gcc", "version": "9.5.0"}
            }
        ]
        
        if self.mode == Mode.TEST:
            matrix["python-version"] = self.PYTHON_VERSIONS
        
        return {"matrix": matrix, "exclude": excludes}
    
    def _generate_gpu_matrix(self) -> Dict[str, Any]:
        """Generate build matrix for GPU variants (CUDA, ROCm, GenAI, HSTU)."""
        host_machines = self._get_filtered_host_machines()
        compilers = self._get_compilers()
        
        matrix = {
            "host-machine": [asdict(hm) for hm in host_machines],
            "compiler": compilers,
        }
        
        if self.mode in [Mode.TEST, Mode.BENCH]:
            matrix["python-version"] = self.PYTHON_VERSIONS
        
        excludes = []
        
        # Add specific exclusions based on variant and mode
        if self.build_variant == BuildVariant.CUDA and self.mode == Mode.BENCH:
            # Reduce matrix size for benchmarks
            matrix["python-version"] = ["3.11"]  # Only one Python version for benchmarks
        
        return {"matrix": matrix, "exclude": excludes}
    
    def _add_mode_specific_config(self, matrix_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add mode-specific configurations to the matrix."""
        if self.mode == Mode.BUILD:
            # Build mode might need specific configurations
            pass
        elif self.mode == Mode.TEST:
            # Test mode configurations
            if "python-version" not in matrix_config["matrix"]:
                matrix_config["matrix"]["python-version"] = self.PYTHON_VERSIONS
        elif self.mode == Mode.BENCH:
            # Benchmark mode - typically reduced matrix for performance
            if "python-version" in matrix_config["matrix"]:
                matrix_config["matrix"]["python-version"] = ["3.11"]  # Single version for benchmarks
        
        return matrix_config
    
    def generate(self) -> str:
        """
        Generate the build matrix as a JSON string.
        
        Returns:
            JSON string representation of the build matrix
        """
        if self.build_variant == BuildVariant.CPU:
            matrix_config = self._generate_cpu_matrix()
        else:
            matrix_config = self._generate_gpu_matrix()
        
        # Apply mode-specific configurations
        matrix_config = self._add_mode_specific_config(matrix_config)
        
        # Add metadata
        matrix_config["_metadata"] = {
            "build_variant": self.build_variant.value,
            "build_target": self.build_target.value,
            "mode": self.mode.value,
        }
        
        return json.dumps(matrix_config, indent=2)
    
    def generate_dict(self) -> Dict[str, Any]:
        """
        Generate the build matrix as a dictionary.
        
        Returns:
            Dictionary representation of the build matrix
        """
        return json.loads(self.generate())


def main():
    """Example usage of the BuildMatrixGenerator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FBGEMM CI build matrix")
    parser.add_argument("--variant", required=True, 
                       choices=[v.value for v in BuildVariant],
                       help="Build variant")
    parser.add_argument("--target", required=True,
                       choices=[t.value for t in BuildTarget], 
                       help="Build target")
    parser.add_argument("--mode", required=True,
                       choices=[m.value for m in Mode],
                       help="Build mode")
    parser.add_argument("--output", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    try:
        generator = BuildMatrixGenerator(args.variant, args.target, args.mode)
        matrix_json = generator.generate()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(matrix_json)
            print(f"Build matrix written to {args.output}")
        else:
            print(matrix_json)
    
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())