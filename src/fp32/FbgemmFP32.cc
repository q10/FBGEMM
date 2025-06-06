/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include <array>
#include <cmath>
#include <utility>

#ifndef __aarch64__
#include "./FbgemmFP32UKernelsAvx2.h" // @manual
#include "./FbgemmFP32UKernelsAvx512.h" // @manual
#include "./FbgemmFP32UKernelsAvx512_256.h" // @manual
#else
#ifdef FBGEMM_ENABLE_KLEIDIAI
#include "./KleidiAIFP32UKernelsNeon.h" // @manual
#endif
#endif
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmFPCommon.h"

namespace fbgemm {

namespace {
// optimized kernels to cover all cases
// 2 in ?x2 should be the same as kernel_ncol_blocks.
// Here with kernel_ncol_blocks = 2, we can provide up to 6x2 kernels, due to
// the restrictions of ymm register numbers (16).
constexpr kernel_array_t<float> kernel_f32_avx2 = {
#ifndef __aarch64__
    nullptr,
    gemmkernel_1x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_2x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_3x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_4x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_5x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_6x2_Avx2_fp32_fA0fB0fC0};
#else
    nullptr};
#endif

constexpr kernel_array_t<float> kernel_f32_avx512 = {
#ifndef __aarch64__
    nullptr,
    gemmkernel_1x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_2x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_3x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_4x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_5x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_6x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_7x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_8x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_9x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_10x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_11x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_12x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_13x2_Avx512_fp32_fA0fB0fC0,
    gemmkernel_14x2_Avx512_fp32_fA0fB0fC0};
#else
    nullptr};
#endif

// clang-format on
constexpr kernel_array_t<float> kernel_f32_avx512_256 = {
#ifndef __aarch64__
    nullptr,
    gemmkernel_1x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_2x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_3x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_4x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_5x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_6x2_Avx2_fp32_fA0fB0fC0,
    gemmkernel_7x2_Avx512_256_fp32_fA0fB0fC0,
    gemmkernel_8x2_Avx512_256_fp32_fA0fB0fC0,
    gemmkernel_9x2_Avx512_256_fp32_fA0fB0fC0,
    gemmkernel_10x2_Avx512_256_fp32_fA0fB0fC0,
    gemmkernel_11x2_Avx512_256_fp32_fA0fB0fC0,
    gemmkernel_12x2_Avx512_256_fp32_fA0fB0fC0,
    gemmkernel_13x2_Avx512_256_fp32_fA0fB0fC0,
    gemmkernel_14x2_Avx512_256_fp32_fA0fB0fC0};
#else
    nullptr};
#endif

#ifdef __aarch64__
#ifdef FBGEMM_ENABLE_KLEIDIAI
constexpr kernel_array_t<float> kernel_fp32_neon = {
    nullptr,
    kleidiai::gemmkernel_1x2_Neon_fp32_fA0fB0fC0,
    kleidiai::gemmkernel_2x2_Neon_fp32_fA0fB0fC0,
    kleidiai::gemmkernel_3x2_Neon_fp32_fA0fB0fC0,
    kleidiai::gemmkernel_4x2_Neon_fp32_fA0fB0fC0,
    kleidiai::gemmkernel_5x2_Neon_fp32_fA0fB0fC0,
    kleidiai::gemmkernel_6x2_Neon_fp32_fA0fB0fC0,
};
#endif
#endif
} // namespace

template <>
const isa_descriptor<float>& getIsaHandlers(inst_set_t isa, float) {
  static isa_descriptor<float> avx2_descriptor =
      std::make_tuple(kernel_f32_avx2, partition_avx2);
  static isa_descriptor<float> avx512_descriptor =
      std::make_tuple(kernel_f32_avx512, partition_avx512);
  static isa_descriptor<float> avx512_256_descriptor =
      std::make_tuple(kernel_f32_avx512_256, partition_avx512);
#ifdef __aarch64__
#ifdef FBGEMM_ENABLE_KLEIDIAI
  static isa_descriptor<float> neon_descriptor =
      std::make_tuple(kernel_fp32_neon, partition_sve128);
#endif
#endif

  switch (isa) {
    case inst_set_t::sve:
#ifdef FBGEMM_ENABLE_KLEIDIAI
      return neon_descriptor;
#endif
    case inst_set_t::anyarch:
    case inst_set_t::avx2:
      return avx2_descriptor;

    case inst_set_t::avx512:
    case inst_set_t::avx512_vnni:
      return avx512_descriptor;

    case inst_set_t::avx512_ymm:
    case inst_set_t::avx512_vnni_ymm:
      return avx512_256_descriptor;
  }

  throw std::runtime_error("Unsupported uArch");
}

#ifdef FBGEMM_FP32_FALLBACK_TO_REF_KERNEL
template <>
FBGEMM_API void ref_kernel<float>(
    int kernel_nrows,
    GemmParams<float>* gp,
    const float* C_base,
    int m_total,
    int n_total,
    int simd_len) {
  int kernel_ncol_blocks = 2;
  int block_col_size = simd_len * kernel_ncol_blocks;
  for (int jb = 0; jb < gp->b_block_cols; ++jb) {
    for (int k = 0; k < gp->k; ++k) {
      for (int i = 0; i < kernel_nrows; ++i) {
        float a = gp->A[i + k * kernel_nrows];
        for (int j = 0; j < block_col_size; ++j) {
          float* C_ptr =
              gp->C + i * (gp->ldc / sizeof(float)) + jb * block_col_size + j;
          assert(C_ptr < C_base + m_total * n_total);
          float b = gp->B[(jb * gp->k + k) * block_col_size + j];
          if (k == 0) {
            if (gp->beta) {
              *C_ptr = std::fma(a, b, (gp->beta) * (*C_ptr));
            } else {
              *C_ptr = a * b;
            }
          } else {
            *C_ptr = std::fma(a, b, *C_ptr);
          }
        }
      }
    }
  }
}
#endif // FBGEMM_FP32_FALLBACK_TO_REF_KERNEL

template void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixB<float>& Bp,
    const float beta,
    float* C,
    int thread_id,
    int num_threads);

} // namespace fbgemm
