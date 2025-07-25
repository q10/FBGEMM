#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import itertools
import sys
from copy import deepcopy
from typing import List

try:
    # pyre-fixme[21]: Could not find name `ArgType` in
    #  `deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.optimizers`.
    # pyre-fixme[21]: Could not find name `OptimItem` in
    #  `deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.optimizers`.
    # pyre-fixme[21]: Could not find name `OptimizerArgsSet` in
    #  `deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.optimizers`.
    # pyre-fixme[21]: Could not find name `generate_optimized_grad_sum_loop_access`
    #  in `deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.optimizers`.
    from .optimizers import *
    from .common import CodeTemplate
    from .optimizer_args import annotation_dict, OptimizerArgsSet
    from .scripts_argsparse import args
except ImportError:
    from optimizers import *

    # pyre-ignore[21]
    from common import CodeTemplate

    # pyre-ignore[21]
    from optimizer_args import annotation_dict

    # pyre-ignore[21]
    from scripts_argsparse import args


class BackwardSplitGenerator:
    @staticmethod
    def render_backward_templates(
        template_filepath: str,
        optimizer: str,
        filename_format: str,
        kwargs: Dict[str, Any],
        is_gwd: bool = False,
    ) -> None:
        if not kwargs.get("has_gpu_support"):
            return

        weighted_options = [True, False]
        nobag_options = [True, False] if (not is_gwd) else [False]
        vbe_options = [True, False] if (kwargs.get("has_vbe_support")) else [False]
        ssd_options = [True, False] if kwargs.get("has_ssd_support") else [False]
        template = CodeTemplate.load(template_filepath)

        for weighted, nobag, vbe, ssd in itertools.product(
            weighted_options, nobag_options, vbe_options, ssd_options
        ):
            if nobag and (weighted or vbe):
                continue
            if kwargs.get("dense") and ssd:
                continue
            if ssd and is_gwd:
                continue

            kdesc = "".join(
                [
                    f"{ 'weighted' if weighted else 'unweighted' }",
                    f"{ '_nobag' if nobag else '' }",
                    f"{ '_vbe' if vbe else '' }",
                ]
            )
            desc = "_".join([f"{ 'ssd' if ssd else 'split' }", kdesc])
            template.write(
                filename_format.format(optimizer, desc),
                weighted=weighted,
                nobag=nobag,
                vbe=vbe,
                is_index_select=False,
                kdesc=kdesc,
                is_gwd=is_gwd,
                ssd=ssd,
                **kwargs,
            )

    @staticmethod
    def generate_backward_split_gpu(**kwargs: Any) -> None:
        """
        Generate CUDA variants of the TBE backward split operators
        """

        optimizer = kwargs.get("optimizer")
        # Generate the backward split kernels
        for template_filepath, filename_format in [
            (
                "training/backward/embedding_backward_split_template.cu",
                "gen_embedding_backward_{}_{}_cuda.cu",
            ),
            (
                "training/backward/embedding_backward_split_meta_template.cpp",
                "gen_embedding_backward_{}_{}_meta.cpp",
            ),
            (
                "training/backward/embedding_backward_split_kernel_cta_template.cu",
                "gen_embedding_backward_{}_{}_kernel_cta.cu",
            ),
            (
                "training/backward/embedding_backward_split_kernel_warp_template.cu",
                "gen_embedding_backward_{}_{}_kernel_warp.cu",
            ),
        ]:
            BackwardSplitGenerator.render_backward_templates(
                template_filepath,
                optimizer,
                filename_format,
                kwargs,
            )

        # Generate the global weight decay CUDA kernels
        if kwargs.get("has_global_weight_decay_support"):
            for template_filepath, filename_format in [
                (
                    "training/backward/embedding_backward_split_kernel_cta_template.cu",
                    "gen_embedding_backward_{}_{}_gwd_kernel_cta.cu",
                ),
                (
                    "training/backward/embedding_backward_split_kernel_warp_template.cu",
                    "gen_embedding_backward_{}_{}_gwd_kernel_warp.cu",
                ),
                (
                    "training/backward/embedding_backward_split_template.cu",
                    "gen_embedding_backward_{}_{}_gwd_cuda.cu",
                ),
            ]:
                BackwardSplitGenerator.render_backward_templates(
                    template_filepath,
                    optimizer,
                    filename_format,
                    kwargs,
                    is_gwd=True,
                )

        for ssd in (
            [True, False]
            if kwargs.get("has_ssd_support") and not kwargs.get("dense")
            else [False]
        ):
            desc = f"{ 'ssd' if ssd else 'split' }"
            # Generate optimizer kernel
            CodeTemplate.load(
                "training/optimizer/embedding_optimizer_split_device_kernel_template.cuh"
            ).write(
                f"gen_embedding_optimizer_{optimizer}_{desc}_device_kernel.cuh",
                ssd=ssd,
                **kwargs,
            )

        # Generate the backward splits
        # We generate only the API to preserve the backward compatibility if
        # has_gpu_support=True
        if not kwargs.get("dense"):
            # Generate CUDA autograd

            # Extract the aux_args and ssd_aux_args for later use
            aux_args = kwargs["aux_args"]
            ssd_aux_args = kwargs["ssd_aux_args"]

            for ssd in [True, False] if kwargs.get("has_ssd_support") else [False]:
                template_filepath = (
                    "training/backward/embedding_backward_split_host_template.cpp"
                )
                desc = "ssd" if ssd else "split"
                sdesc = "_ssd" if ssd else ""
                filename = f"gen_embedding_backward_{desc}_{optimizer}.cpp"
                CodeTemplate.load(template_filepath).write(
                    filename, is_forward=False, ssd=ssd, **kwargs
                )

                # Generate PT2 unified autograd, and PT2 backward wrapper for all optimizers
                for template_filepath, filename in [
                    (
                        "training/pt2/embedding_split_host_pt2_autograd_template.cpp",
                        f"gen_embedding_{desc}_{optimizer}_pt2_autograd.cpp",
                    ),
                    (
                        "training/pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp",
                        f"gen_embedding_backward_{desc}_{optimizer}_pt2_cuda_wrapper.cpp",
                    ),
                ]:
                    CodeTemplate.load(template_filepath).write(
                        filename,
                        is_forward=False,
                        ssd=ssd,
                        schema_annotation=annotation_dict,
                        **kwargs,
                    )

                if kwargs.get("has_cpu_support") or kwargs.get("has_gpu_support"):
                    # Since the template file only uses aux_args, reset the key
                    # based on whether we are generated for SSD variant or not
                    kwargs["aux_args"] = ssd_aux_args if ssd else aux_args

                    # Generates Python invoker for CUDA + CPU, and PT2
                    template = CodeTemplate.load(
                        "training/python/split_embedding_codegen_lookup_invoker.template"
                    )
                    for filename in [
                        f"lookup_{optimizer}{sdesc}.py",
                    ]:
                        template.write(
                            filename, is_fbcode=args.is_fbcode, ssd=ssd, **kwargs
                        )

        else:
            template_filepath = (
                "training/backward/embedding_backward_split_host_template.cpp"
            )
            filename = "gen_embedding_backward_split_dense.cpp"
            CodeTemplate.load(template_filepath).write(
                filename,
                is_forward=False,
                **kwargs,
            )

    @staticmethod
    def generate_backward_split_cpu(**kwargs: Any) -> None:
        """
        Generate CPU variants of the TBE backward split operators
        """

        optimizer = kwargs.get("optimizer")

        # Generate the backward splits
        if kwargs.get("has_cpu_support"):
            CodeTemplate.load(
                "training/backward/embedding_backward_split_cpu_approx_template.cpp"
                if "approx" in optimizer
                else "training/backward/embedding_backward_split_cpu_template.cpp"
            ).write(f"gen_embedding_backward_{optimizer}_split_cpu.cpp", **kwargs)

        # Generate the backward splits (non-dense)
        if not kwargs.get("dense"):
            for template_filepath, filename in [
                (
                    "training/backward/embedding_backward_split_host_cpu_template.cpp",
                    f"gen_embedding_backward_split_{optimizer}_cpu.cpp",
                ),
                (
                    "training/pt2/embedding_split_host_pt2_cpu_wrapper_template.cpp",
                    f"gen_embedding_backward_split_{optimizer}_pt2_cpu_wrapper.cpp",
                ),
            ]:
                CodeTemplate.load(template_filepath).write(
                    filename,
                    is_forward=False,
                    schema_annotation=annotation_dict,
                    **kwargs,
                )

    @staticmethod
    def generate_backward_split(**kwargs: Any) -> None:
        gen_args = kwargs["args"]
        kwargs["args_pt2"] = gen_args.any

        kwargs["args"] = gen_args.cuda
        BackwardSplitGenerator.generate_backward_split_gpu(**kwargs)

        kwargs["args"] = gen_args.cpu
        BackwardSplitGenerator.generate_backward_split_cpu(**kwargs)

    @staticmethod
    def generate_backward_device() -> None:
        # Generate backward device kernels based on weighted (True/False), VBE
        # (True/False), no bag (True/False)
        template_filepath = (
            "training/backward/embedding_backward_split_device_kernel_template.cuh"
        )

        BackwardSplitGenerator.render_backward_templates(
            template_filepath,
            "",
            "{}gen_embedding_backward_{}_device_kernel.cuh",
            {
                "has_gpu_support": True,
                "has_vbe_support": True,
                "has_ssd_support": True,
                "dense": False,
                "gen_once": False,
            },
        )

        # Generate common backward device kernels (generate only once)
        CodeTemplate.load(template_filepath).write(
            "gen_embedding_backward_split_common_device_kernel.cuh",
            gen_once=True,
        )

    @staticmethod
    def generate_backward_grad() -> None:
        # Generate the common grad functions
        CodeTemplate.load(
            "training/backward/embedding_backward_split_grad_template.cu"
        ).write(
            "gen_embedding_backward_split_grad_embedding_ops.cu", is_index_select=False
        )

    @staticmethod
    def generate_backward_indices() -> None:
        template = CodeTemplate.load(
            "training/backward/embedding_backward_split_indice_weights_template.cu"
        )
        dense_options = [True, False]
        ssd_options = [True, False]
        for dense, ssd in itertools.product(dense_options, ssd_options):
            if dense and ssd:
                continue
            desc = "dense" if dense else ("ssd" if ssd else "split")
            template.write(
                f"gen_embedding_backward_{ desc }_indice_weights_codegen_cuda.cu",
                dense=dense,
                ssd=ssd,
            )

    @staticmethod
    def generate_rocm_backward_split(**kwargs: Any) -> None:
        # Generate backward device kernels based on weighted (True/False), VBE
        # (True/False), no bag (True/False)
        template_filepath = (
            "training/backward/rocm/embedding_backward_split_device_kernel_template.hip"
        )

        BackwardSplitGenerator.render_backward_templates(
            template_filepath,
            "",
            "{}gen_embedding_backward_{}_device_kernel_hip.hip",
            {
                "has_gpu_support": True,
                "has_vbe_support": False,
                "has_ssd_support": False,
                "dense": False,
                "gen_once": False,
            },
        )

    @staticmethod
    def generate_backward_header(
        aux_args: Dict[str, List[str]], aux_names: List[str], is_ssd: bool = False
    ) -> None:
        """
        Generate a header file that contains enum of argument order from the dict

        Parameters:
            aux_args (Dict[str, List[str]]): a dict containing a list of arguments
            aux_names (List[str]): names of the argument types (e.g. aux_tensor, aux_int, etc.)
        Return:
            None
        """
        # Generate backward header for PT2 Autograd
        template = CodeTemplate.load("training/pt2/pt2_arg_utils_template.h")
        name_suffix = "_ssd" if is_ssd else ""
        template.write(
            f"pt2_arg_utils{name_suffix}.h", aux_args=aux_args, aux_names=aux_names
        )

    @staticmethod
    def generate_python_sources(
        all_optimizers: List[str], ssd_optimizers: List[str]
    ) -> None:
        CodeTemplate.load("training/python/__init__.template").write(
            "__init__.py", all_optimizers=all_optimizers, ssd_optimizers=ssd_optimizers
        )

        template = CodeTemplate.load("training/python/lookup_args.template")
        for ssd in [True, False]:
            sdesc = "_ssd" if ssd else ""
            filename = f"lookup_args{sdesc}.py"
            template.write(filename, ssd=ssd)

    @staticmethod
    def generate() -> None:
        # Generate backwards and optimizers
        optimizers = [
            dense(),
            adagrad(),
            adam(),
            lamb(),
            lars_sgd(),
            partial_rowwise_adam(),
            partial_rowwise_lamb(),
            rowwise_adagrad(),
            approx_rowwise_adagrad(),
            rowwise_adagrad_with_weight_decay(),
            approx_rowwise_adagrad_with_weight_decay(),
            rowwise_adagrad_with_counter(),
            approx_rowwise_adagrad_with_counter(),
            rowwise_weighted_adagrad(),
            sgd(),
            approx_sgd(),
            none_optimizer(),
        ]

        ssd_tensors = [
            "row_addrs",
            "inserted_rows",
            "post_bwd_evicted_indices",
            "actions_count",
        ]

        aux_names = ["aux_tensor", "aux_int", "aux_float", "aux_bool"]
        # This is a dict of auxilary arguments used in TBE PT2 interface where the aux
        # arguments of a type are packed into a list for that type. This dict maintains the
        # order of the arguments of each type.
        aux_args: Dict[str, List[str]] = {
            "aux_tensor": [
                "B_offsets",  # 0
                "vbe_output_offsets_feature_rank",  # 1
                "vbe_B_offsets_rank_per_feature",  # 2
                "lxu_cache_locations",  # 3
                "uvm_cache_stats",  # 4
                "prev_iter_dev",  # 5
            ],
            "aux_int": [
                "iter",  # 0
                "info_B_num_bits",  # 1
                "info_B_mask",  # 2
            ],
            "aux_float": [
                "gwd_lower_bound",  # 0
                "max_gradient",  # 1
            ],
            "aux_bool": [
                "is_experimental_tbe",  # 0
                "use_uniq_cache_locations_bwd",  # 1
                "use_homogeneous_placements",  # 2
                "apply_global_weight_decay",  # 3
                "gradient_clipping",  # 4
                "stochastic_rounding",  # 5
                "mixed_D",  # 6
            ],
        }

        # SSD-specific arguments
        ssd_aux_bool = [
            # When set to true, the per-row optimizer state will offloaded to
            # the end of each row in the SSD cache.
            "enable_optimizer_offloading",  # 7
        ]

        assert (
            list(aux_args.keys()) == aux_names
        ), f"{aux_names} must match {aux_args.keys()}"

        ssd_aux_args = deepcopy(aux_args)
        ssd_aux_args["aux_bool"].extend(ssd_aux_bool)

        all_optimizers = []
        ssd_optimizers = []

        for optimizer in optimizers:
            optim = optimizer["optimizer"]

            if (
                optimizer["has_cpu_support"] or optimizer["has_gpu_support"]
            ) and optim != "dense":
                all_optimizers.append(optim)
                if optimizer["has_ssd_support"]:
                    ssd_optimizers.append(optim)

            BackwardSplitGenerator.generate_backward_split(
                ssd_tensors=ssd_tensors,
                # Both aux_args and ssd_aux_args will be passed in, since
                # generate_backward_split will generate both SSD and non-SSD
                # variants
                aux_args=aux_args,
                ssd_aux_args=ssd_aux_args,
                **optimizer,
            )

        BackwardSplitGenerator.generate_rocm_backward_split()

        # Generate common device kernels for backwards
        BackwardSplitGenerator.generate_backward_device()

        # Generate forwards and specialized backwards
        BackwardSplitGenerator.generate_backward_grad()
        BackwardSplitGenerator.generate_backward_indices()

        # Generate headers for backwards
        for is_ssd in [True, False]:
            BackwardSplitGenerator.generate_backward_header(
                (ssd_aux_args if is_ssd else aux_args), aux_names, is_ssd=is_ssd
            )

        BackwardSplitGenerator.generate_python_sources(all_optimizers, ssd_optimizers)


def main() -> None:
    BackwardSplitGenerator.generate()


if __name__ == "__main__":
    print(f"[GENERAATE BACKWARD SPLIT]: {sys.argv}")
    main()
