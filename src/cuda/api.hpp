/**
 * @file
 *
 * @brief A single file which includes, in turn, all (joint)
 * wrappers for Runtime and Driver APIs, and related headers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_HPP_
#define CUDA_API_WRAPPERS_HPP_

#include "api/types.hpp"
#include "api/pci_id.hpp"
#include "api/constants.hpp"
#include "api/error.hpp"
#include "api/versions.hpp"
#include "api/miscellany.hpp"
#include "api/pointer.hpp"
#include "api/device_properties.hpp"
#include "api/current_context.hpp"
#include "api/ipc.hpp"
#include "api/array.hpp"
#include "api/texture_view.hpp"
#include "api/copy_parameters.hpp"
#include "api/memory.hpp"
#if CUDA_VERSION >= 11020
#include "api/memory_pool.hpp"
#endif
#include "api/unique_region.hpp"
#include "cuda/api/detail/unique_span.hpp"
#include "api/link_options.hpp"

#include "api/device.hpp"
#include "api/context.hpp"
#include "api/primary_context.hpp"
#include "api/stream.hpp"
#include "api/event.hpp"
#include "api/kernel.hpp"
#include "api/module.hpp"
#if CUDA_VERSION >= 11040
#if !defined(_WIN32) && !defined(WIN32)
#include "api/demangle.hpp"
#endif
#endif
#if CUDA_VERSION >= 12000
#include "api/library.hpp"
#include "api/kernels/in_library.hpp"
#endif
#include "api/link.hpp"

#include "api/current_device.hpp"

#include "api/peer_to_peer.hpp"
#include "api/devices.hpp"

#include "api/detail/pci_id.hpp"
#include "api/kernels/apriori_compiled.hpp"
#include "api/launch_configuration.hpp"
#include "api/kernel_launch.hpp"
#include "api/virtual_memory.hpp"
#if CUDA_VERSION >= 10000
#include "api/external.hpp"
#endif // CUDA_VERSION >= 10000

#if CUDA_VERSION >= 10000
#include "api/graph/node.hpp"
#include "api/graph/typed_node.hpp"
#include "api/graph/template.hpp"
#include "api/graph/instance.hpp"
#endif // CUDA_VERSION >= 10000

#include "api/multi_wrapper_impls/pointer.hpp"
#include "api/multi_wrapper_impls/array.hpp"
#include "api/multi_wrapper_impls/event.hpp"
#include "api/multi_wrapper_impls/device.hpp"
#include "api/multi_wrapper_impls/context.hpp"
#include "api/multi_wrapper_impls/stream.hpp"
#include "api/multi_wrapper_impls/memory.hpp"
#include "api/multi_wrapper_impls/unique_region.hpp"
#include "api/multi_wrapper_impls/unique_span.hpp"
#include "api/multi_wrapper_impls/virtual_memory.hpp"
#include "api/multi_wrapper_impls/kernel.hpp"
#include "api/multi_wrapper_impls/kernel_launch.hpp"
#include "api/multi_wrapper_impls/apriori_compiled_kernel.hpp"
#include "api/multi_wrapper_impls/module.hpp"
#include "api/multi_wrapper_impls/ipc.hpp"
#include "api/multi_wrapper_impls/launch_configuration.hpp"
#include "api/launch_config_builder.hpp"
#if CUDA_VERSION >= 10000
#include "api/multi_wrapper_impls/graph.hpp"
#include "api/graph/node_builder.hpp"
#endif // CUDA_VERSION >= 10000
#endif // CUDA_API_WRAPPERS_HPP_
