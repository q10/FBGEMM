/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

namespace fbgemm_gpu::gen_ai::attention {

std::tuple<at::Tensor, at::Tensor, at::Tensor> gqa_attn_splitk(
    const at::Tensor& XQ,
    const at::Tensor& cache_K,
    const at::Tensor& cache_V,
    const at::Tensor& seq_positions,
    const double qk_scale,
    const int64_t num_split_ks,
    const int64_t kv_cache_quant_num_groups,
    const bool use_tensor_cores,
    const int64_t cache_logical_dtype_int);

at::Tensor mqa_attn(
    at::Tensor XQ,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seq_positions,
    double qk_scale,
    std::optional<int64_t> num_groups,
    int64_t cache_logical_dtype_int,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v);

} // namespace fbgemm_gpu::gen_ai::attention

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl(
      "gqa_attn_splitk",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm_gpu::gen_ai::attention::gqa_attn_splitk)));
  m.impl(
      "mqa_attn",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm_gpu::gen_ai::attention::mqa_attn)));
}
