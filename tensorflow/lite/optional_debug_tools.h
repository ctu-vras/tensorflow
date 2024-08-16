/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
/// \file
///
/// Optional debugging functionality.
/// For small sized binaries, these are not needed.
#ifndef TENSORFLOW_LITE_OPTIONAL_DEBUG_TOOLS_H_
#define TENSORFLOW_LITE_OPTIONAL_DEBUG_TOOLS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"

namespace tflite {
// Returns the name of the allocation type.
const char* AllocTypeName(TfLiteAllocationType type);

// Prints a dump of what tensors and what nodes are in the interpreter.
void PrintInterpreterState(const impl::Interpreter* interpreter,
                           int32_t tensor_name_display_length = 25,
                           int32_t tensor_type_display_length = 15,
                           int32_t alloc_type_display_length = 18);

struct SubgraphDelegationMetadata {
  // A bit vector indicating whether a node is replaced by a delegate.
  std::vector<bool> replaced_node_bits;
  // A vector mapping from the node id of a replaced node to the node id of
  // the delegate node that replaced it.
  std::vector<size_t> replaced_by_node;
  // Whether any delegate has been applied to the subgraph.
  bool has_delegate_applied = false;
};

// Returns the metadata of the delegation of the subgraph.
SubgraphDelegationMetadata GetNodeDelegationMetadata(const Subgraph& subgraph);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_OPTIONAL_DEBUG_TOOLS_H_
