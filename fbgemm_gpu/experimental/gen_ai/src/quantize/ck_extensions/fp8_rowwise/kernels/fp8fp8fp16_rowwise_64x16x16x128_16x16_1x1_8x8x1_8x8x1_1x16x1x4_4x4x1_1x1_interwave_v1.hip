/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_common.h"

at::Tensor
fp8fp8f16_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.
  return f8f8f16_rowwise_wrapper<
      64,
      16,
      16,
      128,
      16,
      16,
      1,
      1,
      S<8, 8, 1>,
      S<8, 8, 1>,
      S<1, 16, 1, 4>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v1>(XQ, WQ, x_scale, w_scale, Y, 1);
}
