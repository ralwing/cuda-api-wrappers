/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */

#include <cuda/api.hpp>
#include <cuda/api/multi_wrapper_impls/kernel_launch.hpp>

#include <Eigen/Eigen>
#include <algorithm>
#include <cuda/std/span>
#include <iostream>
#include <random>

using Vector3f = Eigen::Vector3f;
using DataType = Vector3f;
using SpanType = cuda::std::span<DataType>;

namespace cuda {
template <> struct is_gpu_compatible<DataType> : ::std::true_type {};
} // namespace cuda

__global__ void vectorAdd(const DataType *A, const DataType *B,
                          DataType *result, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    result[i] = A[i] + B[i];
  }
}

__global__ void vectorAddValue(const DataType A, const DataType B,
                               DataType *result) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < 1) {
    *result = A + B;
  }
}

__global__ void vectorAddSpan(const SpanType A, const SpanType B, SpanType C) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < C.size()) {
    C[i] = A[i] + B[i];
  }
}

void print_result(const DataType a, const DataType b, const DataType result) {
  std::cout << "A: " << a.transpose() << " B: " << b.transpose()
            << " Result: " << result.transpose() << ". Should be "
            << (a + b).transpose() << "\n";
}

int main() {
  int numElements = 1000;

  if (cuda::device::count() == 0) {
    std::cerr << "No CUDA devices on this system" << "\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "[Vector addition of " << numElements << " elements]\n";

  auto h_A = std::vector<DataType>(numElements);
  auto h_B = std::vector<DataType>(numElements);
  auto h_result = std::vector<DataType>(numElements);

  auto generator = []() -> DataType {
    static std::random_device random_device;
    static std::mt19937 randomness_generator{random_device()};
    static std::uniform_real_distribution<float> fdistribution{0.0, 1.0};
    static std::uniform_int_distribution<int> idistribution{1, 100};
    DataType result;
    result << fdistribution(randomness_generator),
        fdistribution(randomness_generator),
        fdistribution(randomness_generator);
    return result;
  };

  std::generate(h_A.begin(), h_A.end(), generator);
  std::generate(h_B.begin(), h_B.end(), generator);

  auto device = cuda::device::current::get();

  auto d_A =
      cuda::memory::make_unique_region(device, numElements * sizeof(DataType));
  auto d_B =
      cuda::memory::make_unique_region(device, numElements * sizeof(DataType));
  auto d_result =
      cuda::memory::make_unique_region(device, numElements * sizeof(DataType));
  auto sp_A = d_A.as_span<DataType>();
  auto sp_B = d_B.as_span<DataType>();
  auto sp_result = d_result.as_span<DataType>();

  auto cuda_device = cuda::device::current::get();
  auto stream = cuda_device.create_stream(cuda::stream::sync);

  {
    auto launch_config = cuda::launch_config_builder()
                             .overall_size(numElements)
                             .block_size(256)
                             .build();
    {

      stream.enqueue.copy(sp_A, h_A);
      stream.enqueue.copy(sp_B, h_B);

      stream.enqueue.kernel_launch(vectorAdd, launch_config, sp_A.data(),
                                   sp_B.data(), sp_result.data(), numElements);

      stream.enqueue.copy(h_result, sp_result);

      stream.synchronize();

      print_result(h_A[0], h_B[0], h_result[0]);
    }

    {
      std::generate(h_A.begin(), h_A.end(), generator);
      std::generate(h_B.begin(), h_B.end(), generator);

      stream.enqueue.copy(sp_A, h_A);
      stream.enqueue.copy(sp_B, h_B);

      stream.enqueue.kernel_launch(vectorAddSpan, launch_config, SpanType{sp_A},
                                   SpanType{sp_B}, SpanType{sp_result});

      stream.enqueue.copy(h_result, sp_result);

      stream.synchronize();
      print_result(h_A[0], h_B[0], h_result[0]);
    }
  }
  {
    auto A = DataType{1.0, 2.0, 3.0};
    auto B = DataType{4.0, 5.0, 6.0};

    {
      DataType result;
      // notice only the result is passed by pointer,
      // all other arguments are passed by value from host DataType *d_result;
      DataType *d_result;
      cudaMalloc((void **)&d_result, sizeof(DataType));

      vectorAddValue<<<1, 1>>>(A, B, d_result);
      cudaDeviceSynchronize();

      cudaMemcpy(&result, d_result, sizeof(DataType), cudaMemcpyDeviceToHost);

      print_result(A, B, result);
      cudaFree(d_result);
    }

    {
      auto launch_config =
          cuda::launch_config_builder().overall_size(1).block_size(1).build();

      DataType result;
      DataType *d_result;
      cudaMalloc((void **)&d_result, sizeof(DataType));

      auto d_result_region = cuda::memory::make_unique_region(
          device, numElements * sizeof(DataType));
      auto sp_result = d_result_region.as_span<DataType>();
      stream.enqueue.kernel_launch(vectorAddValue, launch_config, A, B,
                                   d_result);
      stream.enqueue.copy(&result, d_result, sizeof(DataType));
      stream.synchronize();
      print_result(A, B, result);

      cudaFree(d_result);
    }
    {
      auto launch_config =
          cuda::launch_config_builder().overall_size(1).block_size(1).build();

      DataType result;

      auto d_result_region = cuda::memory::make_unique_region(
          device, numElements * sizeof(DataType));
      auto sp_result = d_result_region.as_span<DataType>();
      stream.enqueue.kernel_launch(vectorAddValue, launch_config, A, B,
                                   sp_result.data());
      stream.enqueue.copy(&result, sp_result.data(), sizeof(DataType));
      stream.synchronize();
      print_result(A, B, result);
    }
  }
}
