# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.platform import test


class TfDataServiceGpuTest(
    data_service_test_base.TestBase,
    parameterized.TestCase,
):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              data_transfer_protocol=["grpc", "local"],
              compression=[False, True],
              pinned=[False, True],
          ),
      )
  )
  def test_pinned(self, data_transfer_protocol, compression, pinned):
    cpus = config.list_logical_devices("CPU")
    gpus = config.list_logical_devices("GPU")

    cluster = self.make_test_cluster(num_workers=1)
    with ops.device_v2(cpus[0].name):
      ds = self.make_distributed_range_dataset(
          num_elements=10,
          cluster=cluster,
          data_transfer_protocol=data_transfer_protocol,
          compression=("AUTO" if compression else None),
      )

    def check_pinned(x):
      return control_flow_ops.with_dependencies(
          [gen_experimental_dataset_ops.check_pinned([x])],
          output_tensor=-1,
      )
    ds = ds.map(check_pinned)

    options = options_lib.Options()
    options.experimental_service.pinned = pinned
    ds = ds.with_options(options)

    get_next = lambda: self.evaluate(self.getNext(ds)())
    # TODO(mpcallanan): Support compression and local.
    if gpus and pinned and data_transfer_protocol == "grpc" and not compression:
      get_next()
    else:
      with self.assertRaisesRegex(errors.InvalidArgumentError, "not pinned"):
        get_next()


if __name__ == "__main__":
  test.main()
