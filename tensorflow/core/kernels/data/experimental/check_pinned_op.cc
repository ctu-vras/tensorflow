/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif

namespace tensorflow {
namespace data {
namespace experimental {

class CheckPinnedOp : public OpKernel {
 public:
  explicit CheckPinnedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
#if GOOGLE_CUDA
    unsigned int unused_flags;
    for (size_t i = 0; i < ctx->num_inputs(); ++i) {
      const Tensor& component = ctx->input(i);
      cudaError_t err =
          cudaHostGetFlags(&unused_flags, const_cast<void*>(component.data()));
      if (err != cudaSuccess) {
        ctx->SetStatus(absl::InvalidArgumentError("not pinned"));
        return;
      }
    }
    return;
#endif

    ctx->SetStatus(absl::FailedPreconditionError(
        "to check pinnedness, the op must be built with `--config=cuda`"));
  }
};

REGISTER_KERNEL_BUILDER(Name("CheckPinned").Device(DEVICE_CPU), CheckPinnedOp);
REGISTER_KERNEL_BUILDER(Name("CheckPinned").Device(DEVICE_GPU), CheckPinnedOp);

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
