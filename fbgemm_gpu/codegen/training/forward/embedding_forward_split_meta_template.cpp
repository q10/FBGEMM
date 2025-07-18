/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
// See https://fburl.com/dw9ljh4h
#}

// Companion template is embedding_forward_split_template.cu

{%- set mdesc =  "dense" if dense else ("ssd" if ssd else "split") %}
{%- set wdesc =  "weighted" if weighted else "unweighted" %}
{%- set vdesc = "_vbe" if vbe else "" %}

{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}

////////////////////////////////////////////////////////////////////////////////
// Required for op registrations
////////////////////////////////////////////////////////////////////////////////
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

[[maybe_unused]] static constexpr int32_t kINT8QparamsBytes = 8;

////////////////////////////////////////////////////////////////////////////////
// Kernel Definitions
////////////////////////////////////////////////////////////////////////////////

{%- for nobag in [True, False] %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- if (not nobag or (not weighted and not vbe)) %}
{%- set has_experimental = (not dense and not nobag and not vbe) %}

{%- for is_gwd in ([True, False]
    if is_valid_gwd_config(
        dense,
        nobag,
        vbe,
        is_index_select,
        has_global_weight_decay_support=True,
        ssd=False,
    ) else [False])
%}
{%- set gwddesc = "_gwd" if is_gwd else "" %}
Tensor
{{ mdesc }}_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}{{ gwddesc }}_meta(
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    {%- if not nobag %}
    const Tensor& D_offsets,
    {%- else %}
    const c10::SymInt D,
    {%- endif %}
    {%- if not nobag %}
    const c10::SymInt total_D,
    {%- endif %}
    {%- if not nobag %}
    const c10::SymInt max_D,
    {% endif %}
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
    {%- endif %}
    {%- if not dense %}
    const Tensor& {{ locs_or_addrs_tensor }},
    const Tensor& uvm_cache_stats,
    {%- endif %}
    const int64_t output_dtype,
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const c10::SymInt vbe_output_size,
    const int64_t info_B_num_bits, // int32_t
    const int64_t info_B_mask_int64, // uint32_t
    {%- endif %}
    {%- if is_gwd %}
    const Tensor& hash_size_cumsum,
    const Tensor& prev_iter_dev,
    const Tensor& learning_rate_tensor,
    const double weight_decay,
    const int64_t iter,
    const double gwd_lower_bound,
    {%- endif %}
    const bool is_experimental
) {
    // NB: omitted the device tests TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL
    {%- if not nobag %}
    auto T = D_offsets.sym_numel() - 1;
    {%- else %}
    auto total_L = indices.sym_numel();
    auto T = weights_offsets.sym_numel();
    {%- endif %}
    TORCH_SYM_CHECK(T.sym_gt(0), "");
    // offsets = [B x T  + 1]
    {%- if is_index_select %}
    const auto total_B = num_warps_per_feature * T;
    const auto B = num_warps_per_feature;
    {%- else %}
    const auto total_B = offsets.sym_size(0) - 1;
    const auto B = total_B / T;
    {%- endif %}
    TORCH_SYM_CHECK(B.sym_ge(0), "");
    {%- if not nobag or is_index_select %}
    {%- if not nobag %}
    TORCH_SYM_CHECK(total_D.sym_gt(0), "");
    TORCH_SYM_CHECK((total_D % 4).sym_eq(0), "");
    {%- endif %}
    TORCH_SYM_CHECK(max_D.sym_le({{ max_embedding_dim }}), "");
    {%- elif not is_index_select %}
    TORCH_SYM_CHECK(D.sym_gt(0), "");
    TORCH_SYM_CHECK((D % 4).sym_eq(0), "");
    {%- endif %}
    {%- if vbe %}
    TORCH_SYM_CHECK(vbe_row_output_offsets.sym_numel().sym_eq(total_B), "");
    TENSORS_HAVE_SAME_SYM_NUMEL(vbe_row_output_offsets, vbe_b_t_map);
    TORCH_SYM_CHECK(vbe_output_size.sym_ge(0), "");
    {%- endif %}

    Tensor output;
    // Fix tensor does not have device error for faketensor when all of the weights are undefined tensors.
    auto options = dev_weights.defined() ? dev_weights.options() : at::TensorOptions().device(at::kMeta);
    {%- if nobag %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    {%- if is_index_select %}
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16);

    TORCH_CHECK_GT(fixed_L_per_warp, 0);
    TORCH_CHECK_GT(num_warps_per_feature, 0);
    if (!permute_output_dim_0_1) {
        TORCH_CHECK_GE(output_size, 0);
        TORCH_CHECK_GT(output_offsets.sym_numel(), 0);
    }

    // If permute_output_dim_0_1 is true, output shape is (batch_size * total_D)
    // Else, output shape is (output_size)
    output = at::empty_symint({output_size}, options.dtype(getScalarType(o_dtype)));
    {%- else %}
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);

    c10::SymInt adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * int64_t(kINT8QparamsBytes);
    }

    output = at::empty_symint({total_L, adjusted_D}, options.dtype(getScalarType(o_dtype)));
    {%- endif %}
    {%- else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    
    {%- if vbe %}
    output = at::empty_symint(
        {vbe_output_size},
        options.dtype(getScalarType(o_dtype))
    );
    {%- else %}
    c10::SymInt total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        // TODO: Why is kINT8QparamsBytes a float
        total_adjusted_D += T * int64_t(kINT8QparamsBytes);
    }
    
    output = at::empty_symint(
        {B, total_adjusted_D},
        options.dtype(getScalarType(o_dtype))
    );
    {%- endif %} {#-/* if vbe */#}
    {%- endif %} // if nobag

    return output;
}

////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    // NB: yes cuda here
    {%- set embedding_codegen_forward_op =
        "{}_embedding{}_codegen_forward_{}{}{}_cuda".format(
            mdesc, ndesc, wdesc, vdesc, gwddesc
        )
    %}
    m.impl("{{ embedding_codegen_forward_op }}", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN({{ mdesc }}_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}{{ gwddesc }}_meta)));
}
{%- endfor %} {#-/* for is_gwd */#}
{%- endif %} {#/* if (not nobag or (not weighted and not vbe)) */#}
{%- endfor %} {#-/* for nobag */#}
    // clang-format on
