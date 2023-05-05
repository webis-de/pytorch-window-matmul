#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include "utils.cuh"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define _VOLATILE_
#define BLOCKSIZE 16

template <typename scalar_t>
__device__ void load(
    int thread_y, int thread_x, int b, int accessor_y, int accessor_x,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> accessor,
    _VOLATILE_ scalar_t shared[BLOCKSIZE][BLOCKSIZE])
{
  if (accessor_y >= 0 && accessor_y < accessor.size(1) && accessor_x >= 0 && accessor_x < accessor.size(2))
    shared[thread_y][thread_x] = accessor[b][accessor_y][accessor_x];
  else
    shared[thread_y][thread_x] = 0;
}

template <typename scalar_t>
__device__ void compute_sub(
    _VOLATILE_ scalar_t x_shared[BLOCKSIZE][BLOCKSIZE],
    _VOLATILE_ scalar_t y_shared[BLOCKSIZE][BLOCKSIZE],
    scalar_t &sub)
{
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
#pragma unroll
  for (int block_idx = 0; block_idx < BLOCKSIZE; block_idx++)
    sub += x_shared[thread_x][block_idx] * y_shared[block_idx][thread_y];
}

template <typename scalar_t>
__global__ void window_matmul_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> C_accessor,
    int window_size)
{
  // A: b x m x k
  // B: b x m x k
  // C: b x m x 2w + 1

  // Block index
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int batch_idx = blockIdx.z;

  // Thread index
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;

  // Starting indices of A and B
  int a_m_start = BLOCKSIZE * block_y;
  int b_m_start = a_m_start + BLOCKSIZE * block_x - window_size;
  int a_m = a_m_start + thread_y;
  int b_m = b_m_start + thread_y;

  // ceil (K / BLOCKSIZE)
  int num_blocks = ceil(A_accessor.size(2) / (float)BLOCKSIZE);

  scalar_t c_sub = 0;
  for (int block_idx = 0; block_idx < num_blocks; block_idx++)
  {
    // Shared memory
    __shared__ scalar_t a_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t b_shared[BLOCKSIZE][BLOCKSIZE];

    // Load the matrices into shared memory
    int block = block_idx * BLOCKSIZE;
    int ab_k = block + thread_x;
    load<scalar_t>(thread_y, thread_x, batch_idx, a_m, ab_k, A_accessor, a_shared);
    load<scalar_t>(thread_x, thread_y, batch_idx, b_m, ab_k, B_accessor, b_shared); // Transpose B
    __syncthreads();

    // Compute the partial product
    compute_sub<scalar_t>(a_shared, b_shared, c_sub);
    __syncthreads();
  }

  // Store the result in C
  int c_m = a_m_start + thread_x;
  int c_w = b_m_start + thread_y - c_m + window_size;
  if (c_m >= C_accessor.size(1) || c_w < 0 || c_w >= C_accessor.size(2))
    return;
  C_accessor[batch_idx][c_m][c_w] = c_sub;
}

template <typename scalar_t>
__global__ void unwindow_matmul_kernel_A(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> C_accessor,
    int window_size)
{
  // A: b x m x 2w + 1
  // B: b x m x k
  // C: b x m x k

  // Block index
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int batch_idx = blockIdx.z;

  // Thread index
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;

  // Starting indices of A and B
  int a_m_start = BLOCKSIZE * block_y;
  int b_k_start = BLOCKSIZE * block_x;
  int a_m = a_m_start + thread_y;
  int b_k = b_k_start + thread_x;

  int num_blocks;
  if (window_size < BLOCKSIZE && BLOCKSIZE <= A_accessor.size(1))
    // edge case when window_size < BLOCKSIZE <= m
    num_blocks = 2;
  else
    // ceil (2w+1 / BLOCKSIZE)
    num_blocks = ceil(A_accessor.size(2) / (float)BLOCKSIZE);

  scalar_t c_sub = 0;
  for (int block_idx = 0; block_idx < num_blocks; block_idx++)
  {
    // Shared memory
    __shared__ scalar_t a_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t b_shared[BLOCKSIZE][BLOCKSIZE];

    // Load the matrices from global memory to shared memory
    int block = block_idx * BLOCKSIZE;
    int aw_idx = block + thread_x;
    int bw_idx = block + thread_y;
    int a_w = aw_idx - thread_y;
    int b_m = a_m_start + bw_idx - window_size;

    load<scalar_t>(thread_y, thread_x, batch_idx, a_m, a_w, A_accessor, a_shared);
    load<scalar_t>(thread_y, thread_x, batch_idx, b_m, b_k, B_accessor, b_shared);
    __syncthreads();

    // Compute the partial product
    compute_sub<scalar_t>(a_shared, b_shared, c_sub);
    __syncthreads();
  }

  // Store the result in C
  int c_m = a_m_start + thread_x;
  int c_k = b_k_start + thread_y;
  if (c_m >= C_accessor.size(1) || c_k >= C_accessor.size(2))
    return;
  C_accessor[batch_idx][c_m][c_k] = c_sub;
}

template <typename scalar_t>
__global__ void unwindow_matmul_kernel_B(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> C_accessor,
    int window_size)
{
  // A: b x m x 2w + 1
  // B: b x m x k
  // C: b x m x k

  // Block index
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int batch_idx = blockIdx.z;

  // Thread index
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;

  // Starting indices of A and B
  int a_m_start = BLOCKSIZE * block_y;
  int b_k_start = BLOCKSIZE * block_x;
  // int a_m = a_m_start + thread_y; // needs to use block_idx
  int b_k = b_k_start + thread_x;

  int num_blocks;
  if (window_size < BLOCKSIZE && BLOCKSIZE <= A_accessor.size(1))
    // edge case when window_size < BLOCKSIZE <= m
    num_blocks = 2;
  else
    // ceil (2w+1 / BLOCKSIZE)
    num_blocks = ceil(A_accessor.size(2) / (float)BLOCKSIZE);

  scalar_t c_sub = 0;
  for (int block_idx = 0; block_idx < num_blocks; block_idx++)
  {
    // Shared memory
    __shared__ scalar_t a_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t b_shared[BLOCKSIZE][BLOCKSIZE];

    // Load the matrices from global memory to shared memory
    int block = block_idx * BLOCKSIZE;
    int a_m = a_m_start + thread_y - window_size + block;
    int aw_idx = block + thread_x;
    int bw_idx = block + thread_y;
    int a_w = aw_idx - thread_y + 2 * window_size - block * 2;
    int b_m = a_m_start + bw_idx - window_size;

    load<scalar_t>(thread_x, thread_y, batch_idx, a_m, a_w, A_accessor, a_shared); // transpose A
    load<scalar_t>(thread_y, thread_x, batch_idx, b_m, b_k, B_accessor, b_shared);
    __syncthreads();

    // Compute the partial product
    compute_sub<scalar_t>(a_shared, b_shared, c_sub);
    __syncthreads();
  }

  // Store the result in C
  int c_m = a_m_start + thread_x;
  int c_k = b_k_start + thread_y;
  if (c_m >= C_accessor.size(1) || c_k >= C_accessor.size(2))
    return;
  C_accessor[batch_idx][c_m][c_k] = c_sub;
}

dim3 get_grid(int b, int m, int window_size)
{
  dim3 grid;
  int num_w_blocks, num_m_blocks;
  if (window_size < BLOCKSIZE && BLOCKSIZE <= m)
    num_w_blocks = 2;
  else
    num_w_blocks = ceil((window_size * 2 + 1) / (float)BLOCKSIZE);
  num_m_blocks = ceil(m / (float)BLOCKSIZE);
  grid.x = num_w_blocks;
  grid.y = num_m_blocks;
  grid.z = b;
  return grid;
}

torch::Tensor window_matmul_fw_cuda(torch::Tensor A, torch::Tensor B, int window_size)
{
  CHECK_CUDA(A);
  CHECK_CUDA(B);

  CHECK_INPUT(A.dim() == B.dim());
  CHECK_INPUT(A.size(0) == B.size(0));
  CHECK_INPUT(A.size(1) == B.size(2));
  CHECK_INPUT(A.size(2) == B.size(1));

  // make contiguous
  A = A.contiguous();
  B = B.transpose(-1, -2).contiguous();

  // initialize output
  torch::Tensor C;
  auto sizes = A.sizes().vec();
  sizes[2] = window_size * 2 + 1;
  C = torch::zeros(sizes, A.options());

  // compute grid
  dim3 grid = get_grid(A.size(0), A.size(1), window_size);
  dim3 threads(BLOCKSIZE, BLOCKSIZE);

  // run kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      C.scalar_type(), "window_matmul_fw_cuda", [&]
      {
        auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto C_accessor = C.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        window_matmul_kernel<scalar_t><<<grid, threads>>>(
          A_accessor,
          B_accessor,
          C_accessor,
          window_size
          ); });
  return C;
}

std::tuple<torch::Tensor, torch::Tensor> window_matmul_bw_cuda(
    torch::Tensor A, torch::Tensor B, int window_size, torch::Tensor grad_C)
{
  CHECK_CUDA(A);
  CHECK_CUDA(B);

  CHECK_INPUT(A.dim() == B.dim());
  CHECK_INPUT(A.size(0) == B.size(0));
  CHECK_INPUT(A.size(1) == B.size(2));
  CHECK_INPUT(A.size(2) == B.size(1));

  // make contiguous
  A = A.contiguous();
  B = B.transpose(-1, -2).contiguous();
  grad_C = grad_C.contiguous();

  // initialize output
  torch::Tensor grad_A, grad_B;
  grad_A = torch::zeros(A.sizes().vec(), A.options());
  grad_B = torch::zeros(B.sizes().vec(), B.options());

  // compute grid
  dim3 grid = get_grid(A.size(0), A.size(1), window_size);
  dim3 threads(BLOCKSIZE, BLOCKSIZE);

  // run kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_C.scalar_type(), "window_matmul_bw_cuda", [&]
      {
        auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_C_accessor = grad_C.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_A_accessor = grad_A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_B_accessor = grad_B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        unwindow_matmul_kernel_A<scalar_t><<<grid, threads>>>(
          grad_C_accessor,
          B_accessor,
          grad_A_accessor,
          window_size);
        unwindow_matmul_kernel_B<scalar_t><<<grid, threads>>>(
          grad_C_accessor,
          A_accessor,
          grad_B_accessor,
          window_size); });
  return std::make_tuple(grad_A, grad_B.transpose(-1, -2));
}

torch::Tensor unwindow_matmul_fw_cuda(torch::Tensor A, torch::Tensor B, int window_size)
{
  CHECK_CUDA(A);
  CHECK_CUDA(B);

  CHECK_INPUT(A.dim() == B.dim());
  CHECK_INPUT(A.size(0) == B.size(0));
  CHECK_INPUT(A.size(1) == B.size(1));
  CHECK_INPUT(A.size(2) == window_size * 2 + 1);

  // make contiguous
  A = A.contiguous();
  B = B.contiguous();

  // initialize output
  torch::Tensor C;
  auto sizes = B.sizes().vec();
  C = torch::zeros(sizes, A.options());

  // compute grid
  dim3 grid = get_grid(A.size(0), A.size(1), window_size);
  dim3 threads(BLOCKSIZE, BLOCKSIZE);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      C.scalar_type(), "unwindow_matmul_fw_cuda", [&]
      {
        auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto C_accessor = C.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        unwindow_matmul_kernel_A<scalar_t><<<grid, threads>>>(
          A_accessor,
          B_accessor,
          C_accessor,
          window_size); });
  return C;
}

std::tuple<torch::Tensor, torch::Tensor> unwindow_matmul_bw_cuda(
    torch::Tensor A, torch::Tensor B, int window_size, torch::Tensor grad_output)
{
  CHECK_CUDA(A);
  CHECK_CUDA(B);

  CHECK_INPUT(A.dim() == B.dim());
  CHECK_INPUT(A.size(0) == B.size(0));
  CHECK_INPUT(A.size(1) == B.size(1));
  CHECK_INPUT(A.size(2) == window_size * 2 + 1);

  A = A.contiguous();
  B = B.contiguous();
  grad_output = grad_output.contiguous();

  // initialize output
  torch::Tensor grad_A, grad_B;
  grad_A = torch::zeros(A.sizes().vec(), A.options());
  grad_B = torch::zeros(B.sizes().vec(), B.options());

  // compute grid
  dim3 grid = get_grid(A.size(0), A.size(1), window_size);
  dim3 threads(BLOCKSIZE, BLOCKSIZE);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_A.scalar_type(), "unwindow_matmul_bw_cuda", [&]
      {
        auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_C_accessor = grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_A_accessor = grad_A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_B_accessor = grad_B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        window_matmul_kernel<scalar_t><<<grid, threads>>>(
          grad_C_accessor,
          B_accessor,
          grad_A_accessor,
          window_size
          );
        unwindow_matmul_kernel_B<scalar_t><<<grid, threads>>>(
          A_accessor,
          grad_C_accessor,
          grad_B_accessor,
          window_size
          ); });
  return std::make_tuple(grad_A, grad_B);
}
