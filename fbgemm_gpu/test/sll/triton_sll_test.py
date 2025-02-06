# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import fbgemm_gpu
import fbgemm_gpu.sll.cpu_sll  # noqa F401
import fbgemm_gpu.sll.triton_sll  # noqa F401
import torch
from hypothesis import given, settings, strategies as st
from torch.testing._internal.optests import opcheck

from .common import open_source  # noqa

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, running_on_rocm
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_rocm


class TritonSLLTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        device_type=st.sampled_from(["cpu", "cuda"]),
        enable_pt2=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_dense_jagged_cat_jagged_out(
        self,
        B: int,
        max_L: int,
        device_type: str,
        enable_pt2: bool,
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, max_L + 1, (B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        c_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths + 1)
        a = torch.randint(0, 100000000, (B,), device=device)
        b = torch.randint(0, 100000000, (int(lengths.sum().item()),), device=device)

        ref = torch.cat(
            [
                (
                    torch.cat((a[i : i + 1], b[offsets[i] : offsets[i + 1]]), dim=-1)
                    if lengths[i] > 0
                    else a[i : i + 1]
                )
                for i in range(B)
            ],
            dim=-1,
        )

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def model(a, b, offsets, max_L):
            return torch.ops.fbgemm.sll_dense_jagged_cat_jagged_out(
                a, b, offsets, max_L
            )

        if enable_pt2:
            model = torch.compile(model)

        ret, c_offsets_computed = model(a, b, offsets, max_L)

        assert torch.allclose(ref, ret)
        assert torch.equal(c_offsets, c_offsets_computed)

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_on_rocm)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 100),
        L=st.integers(1, 200),
        device_type=st.sampled_from(["cpu", "cuda"]),
        enable_pt2=st.booleans(),
    )
    @settings(deadline=None)
    def test_triton_jagged_self_substraction_jagged_out(
        self,
        B: int,
        L: int,
        device_type: str,
        enable_pt2: bool,
    ) -> None:
        device = torch.device(device_type)
        torch.manual_seed(0)
        lengths_a = torch.randint(1, L + 2, (B,), device=device)
        offsets_a = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths_a.cumsum(dim=0),
            ],
            dim=0,
        )

        lengths_b = (lengths_a - 1) * (lengths_a - 1)

        offsets_b = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths_b.cumsum(dim=0),
            ],
            dim=0,
        )

        jagged_A = torch.randint(
            0, 100000000, (int(lengths_a.sum().item()),), device=device
        )

        def model(
            jagged_A: torch.Tensor,
            offsets_a: torch.Tensor,
            offsets_b: torch.Tensor,
            L: int,
        ) -> torch.Tensor:
            return torch.ops.fbgemm.sll_jagged_self_substraction_jagged_out(
                jagged_A,
                offsets_a,
                offsets_b,
                L,
            )

        if enable_pt2:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True
            opcheck(
                torch.ops.fbgemm.sll_jagged_self_substraction_jagged_out,
                (jagged_A, offsets_a, offsets_b, L),
            )
            model = torch.compile(model)

        result = model(jagged_A, offsets_a, offsets_b, L)

        for i in range(B):
            if lengths_a[i] == 1:
                continue

            a = jagged_A[offsets_a[i] : offsets_a[i + 1]]
            ref = a[:-1].unsqueeze(1) - a[1:].unsqueeze(0)

            assert torch.equal(result[offsets_b[i] : offsets_b[i + 1]], ref.flatten())
