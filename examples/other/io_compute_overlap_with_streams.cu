/**
 * An example program utilizing most/all calls from the CUDA
 * Runtime API module:
 *
 *   Stream Management
 *
 */
#include <cuda/api.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>

using element_t = float;

using clock_value_t = long long;

__device__ void gpu_sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; }
    while (cycles_elapsed < sleep_cycles);
}

template <typename T>
__global__ void add(
    const T* __restrict__  lhs,
    const T* __restrict__  rhs,
    T* __restrict__        result,
    size_t                 length)
{
    auto global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index < length) {
        result[global_index] = lhs[global_index] + rhs[global_index];
        gpu_sleep(200000);
    }
}

template <typename I, typename I2>
constexpr I div_rounding_up(I dividend, const I2 divisor) noexcept
{
    return (dividend / divisor) + !!(dividend % divisor);
}

struct buffer_set_t {
    cuda::unique_span<element_t> host_lhs;
    cuda::unique_span<element_t> host_rhs;
    cuda::unique_span<element_t> host_result;
    cuda::unique_span<element_t> device_lhs;
    cuda::unique_span<element_t> device_rhs;
    cuda::unique_span<element_t> device_result;
};

std::vector<buffer_set_t> generate_buffers(
    const cuda::device_t&  device,
    size_t                num_kernels,
    size_t                num_elements)
{
    // device.make_current();
    // TODO: This should be an std::array, but generating
    // it is a bit tricky and I don't want to burden the example
    // with template wizardry
    std::vector<buffer_set_t> buffers;
    std::generate_n(std::back_inserter(buffers), num_kernels,
        [&]() {
            return buffer_set_t {
                // Sticking to C++11 here...
                cuda::memory::host::make_unique_span<element_t>(num_elements),
                cuda::memory::host::make_unique_span<element_t>(num_elements),
                cuda::memory::host::make_unique_span<element_t>(num_elements),
                cuda::memory::make_unique_span<element_t>(device, num_elements),
                cuda::memory::make_unique_span<element_t>(device, num_elements),
                cuda::memory::make_unique_span<element_t>(device, num_elements)
            };
        }
    );

    // TODO: Consider actually filling the buffers

    return buffers;
}

int main(int, char **)
{
    constexpr size_t num_kernels     = 5;
    constexpr size_t num_elements    = 10 * 1000 * 1000;

    auto device = cuda::device::current::get();
    std::cout << "Using CUDA device " << device.name() << " (having ID " << device.id() << ")\n";

    std::cout << "Generating host buffers... " << std::flush;
    auto buffers = generate_buffers(device, num_kernels, num_elements);
    std::cout << "done.\n" << std::flush;

    std::vector<cuda::stream_t> streams;
    streams.reserve(num_kernels);
    std::generate_n(std::back_inserter(streams), num_kernels,
        [&]() { return device.create_stream(cuda::stream::async); });

    auto common_launch_config = cuda::launch_config_builder()
		.device(device)
		.overall_size(num_elements)
		.use_maximum_linear_block()
		.build();


	auto buffer_size = num_elements * sizeof(element_t);

    std::cout
        << "Running " << num_kernels << " sequences of HtoD-kernel-DtoH, in parallel" << std::endl;
        // Unfortunately, we need to use indices here - unless we
        // had access to a zip iterator (e.g. boost::zip_iterator)
    for(size_t k = 0; k < num_kernels; k++) {
        auto& stream = streams[k];
        auto& buffer_set = buffers[k];
        stream.enqueue.copy(buffer_set.device_lhs, buffer_set.host_lhs);
        stream.enqueue.copy(buffer_set.device_rhs, buffer_set.host_rhs);
        stream.enqueue.kernel_launch(
            add<element_t>,
            common_launch_config,
            buffer_set.device_lhs.data(),
            buffer_set.device_rhs.data(),
            buffer_set.device_result.data(),
            num_elements);
        stream.enqueue.copy(buffer_set.host_result.get(), buffer_set.device_result.get(), buffer_size);
	auto callback = [=] {
		std::cout << "Stream " << k+1 << " of " << num_kernels << " has concluded all work. " << std::endl;
	};
        stream.enqueue.host_invokable(callback);
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50000));
    for(auto& stream : streams) { stream.synchronize(); }
    cuda::outstanding_error::ensure_none();

    // TODO: Consider checking for correctness here

    std::cout << "\nSUCCESS" << std::endl;
}
