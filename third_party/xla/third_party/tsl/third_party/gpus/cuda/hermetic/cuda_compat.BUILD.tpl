licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "cuda_shared_library",
    shared_library = "lib/libcuda.so",
)
cc_import(
    name = "nvidia_nvvm_shared_library",
    shared_library = "lib/libnvidia-nvvm.so.%{libcuda_version}",
)
cc_import(
    name = "nvidia_pkcs11_openssl3_shared_library",
    shared_library = "lib/libnvidia-pkcs11-openssl3.so.%{libcuda_version}",
)
cc_import(
    name = "nvidia_pkcs11_shared_library",
    shared_library = "lib/libnvidia-pkcs11-openssl3.so.%{libcuda_version}",
)
cc_import(
    name = "nvidia_ptxjitcompiler_shared_library",
    shared_library = "lib/libnvidia-ptxjitcompiler.so.%{libcuda_version}",
)
%{multiline_comment}
cc_library(
    name = "cuda_compat",
    deps = [
      %{comment}":cuda_shared_library",
      %{comment}":nvidia_nvvm_shared_library",
      %{comment}":nvidia_pkcs11_shared_library",
      %{comment}":nvidia_pkcs11_openssl3_shared_library",
      %{comment}":nvidia_ptxjitcompiler_openssl3_shared_library",
    ],
    visibility = [
      "@cuda_cudart//:__pkg__",
      "@local_config_cuda//cuda:__pkg__"],
)