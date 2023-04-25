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
    int thread_x, int thread_y, int b, int accessor_x, int accessor_y,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> accessor,
    _VOLATILE_ scalar_t shared[BLOCKSIZE][BLOCKSIZE])
{
  if (
      accessor_x >= 0 && accessor_x < accessor.size(1) && accessor_y >= 0 && accessor_y < accessor.size(2))
  {
    shared[thread_y][thread_x] = accessor[b][accessor_x][accessor_y];
  }
  else
  {
    shared[thread_y][thread_x] = 0;
  }
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
  {
    sub += x_shared[thread_x][block_idx] * y_shared[block_idx][thread_y];
  }
}

template <typename scalar_t>
__global__ void window_matmul_fw_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> C_accessor,
    int window_size)
{
  // A: b x m x k
  // B: b x k x m
  // C: b x m x 2w + 1
  // grid dim: ceil((2w + 1) / blocksize), ceil(m / blocksize), b

  // Block index
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int batch_idx = blockIdx.z;

  // Starting indices of A and B
  int a_m_start = BLOCKSIZE * block_y;
  int b_m_start = a_m_start + BLOCKSIZE * block_x - window_size;

  // Thread index
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
  int b_m = b_m_start + thread_x;
  int a_m = a_m_start + thread_y;

  // ceil (K / BLOCKSIZE)
  int num_blocks = (A_accessor.size(2) + BLOCKSIZE - 1) / BLOCKSIZE;

  scalar_t c_sub = 0;
  for (int block_idx = 0; block_idx < num_blocks; block_idx++)
  {
    // Shared memory
    __shared__ scalar_t a_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t b_shared[BLOCKSIZE][BLOCKSIZE];

    // Load the matrices into shared memory
    int a_k = block_idx * BLOCKSIZE + thread_x;
    int b_k = block_idx * BLOCKSIZE + thread_y;
    load<scalar_t>(thread_x, thread_y, batch_idx, a_m, a_k, A_accessor, a_shared);
    load<scalar_t>(thread_x, thread_y, batch_idx, b_k, b_m, B_accessor, b_shared);
    __syncthreads();

    // Compute the partial product
    compute_sub<scalar_t>(a_shared, b_shared, c_sub);
    __syncthreads();
  }

  // Store the result in C
  int c_x = a_m_start + thread_x;
  int c_y = b_m_start + thread_y - c_x + window_size;
  if (c_x >= 0 && c_x < C_accessor.size(1) && c_y >= 0 && c_y < C_accessor.size(2))
  {
    C_accessor[batch_idx][c_x][c_y] = c_sub;
  }
}

template <typename scalar_t>
__global__ void window_matmul_bw_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_C_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_B_accessor,
    int window_size)
{
  // A: b x m x k
  // B: b x k x m
  // C: b x m x 2w + 1
  // grid dim: ceil(k / blocksize), ceil(m / blocksize), b

  // Block index
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int batch_idx = blockIdx.z;

  // Starting and ending indices of A, B and C
  int ab_k_start = BLOCKSIZE * block_x;
  int c_m_start = BLOCKSIZE * block_y;

  // Thread index
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
  int a_k = ab_k_start + thread_x;
  int b_k = ab_k_start + thread_y;
  int c_m = c_m_start + thread_y;

  int num_blocks;
  if (window_size < BLOCKSIZE && BLOCKSIZE <= A_accessor.size(1))
    // edge case when window_size < BLOCKSIZE <= m
    num_blocks = 2;
  else
    // ceil (2w+1 / BLOCKSIZE)
    num_blocks = (A_accessor.size(2) + BLOCKSIZE - 1) / BLOCKSIZE;

  scalar_t a_sub = 0;
  scalar_t b_sub = 0;
  for (int block_idx = 0; block_idx < num_blocks; block_idx++)
  {
    // Shared memory
    __shared__ scalar_t a_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t b_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t c_shared[BLOCKSIZE][BLOCKSIZE];

    // Load the matrices into shared memory
    int aw_idx = block_idx * BLOCKSIZE + thread_y;
    int bw_idx = block_idx * BLOCKSIZE + thread_x;
    int cw_idx = block_idx * BLOCKSIZE + thread_x;
    int a_m = c_m_start + aw_idx - window_size;
    int b_m = c_m_start + bw_idx - window_size;
    int c_w = cw_idx - thread_y;

    load<scalar_t>(thread_x, thread_y, batch_idx, a_m, a_k, A_accessor, a_shared);
    load<scalar_t>(thread_y, thread_x, batch_idx, b_k, b_m, B_accessor, b_shared);
    load<scalar_t>(thread_x, thread_y, batch_idx, c_m, c_w, grad_C_accessor, c_shared);
    __syncthreads();

    // Compute the partial product
    compute_sub<scalar_t>(b_shared, c_shared, a_sub);
    compute_sub<scalar_t>(a_shared, c_shared, b_sub);
    __syncthreads();
  }

  // Store the result in grad_A and grad_B
  int ab_m = c_m_start + thread_y;
  int ab_k = ab_k_start + thread_x;
  if (ab_m < A_accessor.size(1) && ab_k < A_accessor.size(2))
  {
    grad_A_accessor[batch_idx][ab_m][ab_k] = a_sub;
    grad_B_accessor[batch_idx][ab_m][ab_k] = b_sub;
  }
}

template <typename scalar_t>
__global__ void unwindow_matmul_fw_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> C_accessor,
    int window_size)
{
  // A: b x m x 2w + 1
  // B: b x m x k
  // C: b x m x k
  // grid dim: ceil(k / blocksize), ceil(m / blocksize), b

  // Block index
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int batch_idx = blockIdx.z;

  // Starting indices of A and B
  int a_m_start = BLOCKSIZE * block_y;
  int b_k_start = BLOCKSIZE * block_x;

  // Thread index
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
  int a_m = a_m_start + thread_y;
  int b_k = b_k_start + thread_x;

  int num_blocks;
  if (window_size < BLOCKSIZE && BLOCKSIZE <= A_accessor.size(1))
    // edge case when window_size < BLOCKSIZE <= m
    num_blocks = 2;
  else
    // ceil (2w+1 / BLOCKSIZE)
    num_blocks = (A_accessor.size(2) + BLOCKSIZE - 1) / BLOCKSIZE;

  scalar_t c_sub = 0;
  for (int block_idx = 0; block_idx < num_blocks; block_idx++)
  {
    // Shared memory
    __shared__ scalar_t a_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t b_shared[BLOCKSIZE][BLOCKSIZE];

    // Load the matrices from global memory to shared memory
    int aw_idx = block_idx * BLOCKSIZE + thread_x;
    int bw_idx = block_idx * BLOCKSIZE + thread_y;
    int a_w = aw_idx - thread_y;
    int b_m = a_m_start + bw_idx - window_size;

    load<scalar_t>(thread_x, thread_y, batch_idx, a_m, a_w, A_accessor, a_shared);
    load<scalar_t>(thread_x, thread_y, batch_idx, b_m, b_k, B_accessor, b_shared);
    __syncthreads();

    // Compute the partial product
    compute_sub<scalar_t>(a_shared, b_shared, c_sub);
    __syncthreads();
  }

  // Store the result in C
  int c_m = a_m_start + thread_x;
  int c_k = b_k_start + thread_y;
  if (c_m < C_accessor.size(1) && c_k < C_accessor.size(2))
  {
    C_accessor[batch_idx][c_m][c_k] = c_sub;
  }
}

template <typename scalar_t>
__global__ void unwindow_matmul_bw_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_C_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_A_accessor,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_B_accessor,
    int window_size)
{
}

// template <typename scalar_t>
// __global__ void A_window_matmul_bw_cuda_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_C_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_A_accessor,
//     int window_size,
//     int b,
//     int m,
//     int k)
// {
//   int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int h, s, b, w_idx;
//   if (thread_idx >= b * m * k)
//     return;
//   b = thread_idx / k / m % b;
//   s = thread_idx / k % m;
//   h = thread_idx % k;
//   scalar_t result = 0;
//   for (auto w = std::max(0, s - window_size); w < std::min(m, s + window_size + 1); w++)
//   {
//     w_idx = w - std::min(0, s - window_size) - std::max(0, s - window_size);
//     result += B_accessor[b][h][w] * grad_C_accessor[b][s][w_idx];
//   }
//   grad_A_accessor[b][s][h] = result;
// }

// template <typename scalar_t>
// __global__ void B_window_matmul_bw_cuda_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_C_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_B_accessor,
//     int window_size,
//     int b,
//     int m,
//     int k)
// {
//   int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int h, s, b, w_idx;
//   if (thread_idx >= b * m * k)
//     return;
//   b = thread_idx / k / m % b;
//   s = thread_idx / k % m;
//   h = thread_idx % k;
//   scalar_t result = 0;
//   for (auto w = std::max(0, s - window_size); w < std::min(m, s + window_size + 1); w++)
//   {
//     w_idx = 2 * window_size - (w - std::min(0, s - window_size) - std::max(0, s - window_size));
//     result += A_accessor[b][w][h] * grad_C_accessor[b][w][w_idx];
//   }
//   grad_B_accessor[b][h][s] = result;
// }

// template <typename scalar_t>
// __global__ void A_unwindow_matmul_bw_cuda_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> B_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_C_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_A_accessor,
//     int window_size,
//     int b,
//     int m,
//     int k)
// {
//   int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int w, s, b, w_idx, full_window_size;
//   full_window_size = window_size * 2 + 1;
//   if (thread_idx >= b * m * full_window_size)
//     return;
//   b = thread_idx / full_window_size / m % b;
//   s = thread_idx / full_window_size % m;
//   w = s - window_size + (thread_idx % full_window_size);
//   if (w < 0 || w >= m)
//     return;
//   w_idx = w - std::min(0, s - window_size) - std::max(0, s - window_size);
//   scalar_t result = 0;
//   for (int h = 0; h < k; h++)
//   {
//     result += B_accessor[b][w][h] * grad_C_accessor[b][s][h];
//   }
//   grad_A_accessor[b][s][w_idx] = result;
// }

// template <typename scalar_t>
// __global__ void B_unwindow_matmul_bw_cuda_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_C_accessor,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_B_accessor,
//     int window_size,
//     int b,
//     int m,
//     int k)
// {
//   int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int h, s, b, w_idx;
//   if (thread_idx >= b * m * k)
//     return;
//   b = thread_idx / k / m % b;
//   s = thread_idx / k % m;
//   h = thread_idx % k;
//   scalar_t result = 0;
//   for (auto w = std::max(0, s - window_size); w < std::min(m, s + window_size + 1); w++)
//   {
//     w_idx = 2 * window_size - (w - std::min(0, s - window_size) - std::max(0, s - window_size));
//     result += A_accessor[b][w][w_idx] * grad_C_accessor[b][w][h];
//   }
//   grad_B_accessor[b][s][h] = result;
// }

torch::Tensor window_matmul_fw_cuda(torch::Tensor A, torch::Tensor B, int window_size)
{
  CHECK_CUDA(A);
  CHECK_CUDA(B);

  CHECK_INPUT(A.dim() == B.dim());
  CHECK_INPUT(A.size(0) == A.size(0));
  CHECK_INPUT(A.size(-1) == B.size(-2));
  CHECK_INPUT(A.size(-2) == B.size(-1));

  A = A.contiguous();
  B = B.contiguous();

  torch::Tensor C;
  auto sizes = A.sizes().vec();
  sizes[2] = window_size * 2 + 1;
  C = torch::zeros(sizes, A.options());
  int b, m, num_m_blocks, num_w_blocks, full_window_size;
  full_window_size = window_size * 2 + 1;
  b = A.size(0);
  m = A.size(1);

  num_m_blocks = (m + BLOCKSIZE - 1) / BLOCKSIZE;
  if (window_size < BLOCKSIZE && BLOCKSIZE <= A.size(1))
    num_w_blocks = 2;
  else
    num_w_blocks = (full_window_size + BLOCKSIZE - 1) / BLOCKSIZE;

  dim3 threads(BLOCKSIZE, BLOCKSIZE);
  dim3 grid(num_w_blocks, num_m_blocks, b);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      C.scalar_type(), "window_matmul_fw_cuda", [&]
      {
        auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto C_accessor = C.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        window_matmul_fw_cuda_kernel<scalar_t><<<grid, threads>>>(
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
  CHECK_INPUT(A.size(0) == A.size(0));
  CHECK_INPUT(A.size(-1) == B.size(-2));
  CHECK_INPUT(A.size(-2) == B.size(-1));

  A = A.contiguous();
  B = B.contiguous();
  grad_C = grad_C.contiguous();

  torch::Tensor grad_A, grad_B;
  grad_A = torch::zeros(A.sizes().vec(), A.options());
  grad_B = torch::zeros(B.sizes().vec(), B.options());
  int b, m, k;
  b = A.size(0);
  m = A.size(1);
  k = A.size(2);

  int num_k_blocks = (k + BLOCKSIZE - 1) / BLOCKSIZE;
  int num_m_blocks = (m + BLOCKSIZE - 1) / BLOCKSIZE;

  dim3 threads(BLOCKSIZE, BLOCKSIZE);
  dim3 grid(num_k_blocks, num_m_blocks, b);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_C.scalar_type(), "window_matmul_bw_cuda", [&]
      {
        auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_C_accessor = grad_C.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_A_accessor = grad_A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_B_accessor = grad_B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        window_matmul_bw_cuda_kernel<scalar_t><<<grid, threads>>>(
          A_accessor,
          B_accessor,
          grad_C_accessor,
          grad_A_accessor,
          grad_B_accessor,
          window_size); });
  return std::make_tuple(grad_A, grad_B);
}

torch::Tensor unwindow_matmul_fw_cuda(torch::Tensor A, torch::Tensor B, int window_size)
{
  CHECK_CUDA(A);
  CHECK_CUDA(B);

  CHECK_INPUT(A.dim() == B.dim());
  CHECK_INPUT(A.size(0) == A.size(0));
  CHECK_INPUT(A.size(-1) == window_size * 2 + 1);
  CHECK_INPUT(A.size(-2) == B.size(-2));

  A = A.contiguous();
  B = B.contiguous();

  torch::Tensor C;
  auto sizes = B.sizes().vec();
  C = torch::zeros(sizes, A.options());
  int b, m, k, num_m_blocks, num_k_blocks, full_window_size;
  b = B.size(0);
  m = B.size(1);
  k = B.size(2);
  num_m_blocks = (m + BLOCKSIZE - 1) / BLOCKSIZE;
  num_k_blocks = (k + BLOCKSIZE - 1) / BLOCKSIZE;

  dim3 threads(BLOCKSIZE, BLOCKSIZE);
  dim3 grid(num_k_blocks, num_m_blocks, b);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      C.scalar_type(), "unwindow_matmul_fw_cuda", [&]
      {
        auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto C_accessor = C.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        unwindow_matmul_fw_cuda_kernel<scalar_t><<<grid, threads>>>(
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
  CHECK_INPUT(A.size(0) == A.size(0));
  CHECK_INPUT(A.size(-1) == window_size * 2 + 1);
  CHECK_INPUT(A.size(-2) == B.size(-2));

  A = A.contiguous();
  B = B.contiguous();
  grad_output = grad_output.contiguous();

  torch::Tensor grad_A, grad_B;
  grad_A = torch::zeros(A.sizes().vec(), A.options());
  grad_B = torch::zeros(B.sizes().vec(), B.options());
  int b, m, k;
  b = B.size(0);
  m = B.size(1);
  k = B.size(2);

  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
  //     grad_A.scalar_type(), "unwindow_matmul_bw_cuda", [&]
  //     {
  //       auto A_accessor = A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
  //       auto B_accessor = B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
  //       auto grad_C_accessor = grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
  //       auto grad_A_accessor = grad_A.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
  //       auto grad_B_accessor = grad_B.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

  //       A_unwindow_matmul_bw_cuda_kernel<scalar_t><<<BLOCKS(b*m*(window_size * 2 + 1)),THREADS>>>(
  //         B_accessor,
  //         grad_C_accessor,
  //         grad_A_accessor,
  //         window_size,
  //         b,
  //         m,
  //         k
  //         );
  //       B_unwindow_matmul_bw_cuda_kernel<scalar_t><<<BLOCKS(b*m*k),THREADS>>>(
  //         A_accessor,
  //         grad_C_accessor,
  //         grad_B_accessor,
  //         window_size,
  //         b,
  //         m,
  //         k
  //         ); });
  return std::make_tuple(grad_A, grad_B);
}
