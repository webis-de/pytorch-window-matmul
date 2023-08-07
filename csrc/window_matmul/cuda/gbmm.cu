/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cute/tensor.hpp>

template <class BHShape, class MShape, class KShape, class FWShape,
          class TA, class AStride, class ABlockLayout, class AThreadLayout,
          class TB, class BStride, class BBlockLayout, class BThreadLayout,
          class TC, class CStride, class CBlockLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void gbmm_nt_nn_device(BHShape BH, MShape M, KShape K, FWShape FW,
                                                                                                   TA const *A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
                                                                                                   TB const *B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
                                                                                                   TC *C, CStride dC, CBlockLayout, CThreadLayout tC)
{
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

    CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
    CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

    // CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));      // BLK_M
    // CUTE_STATIC_ASSERT_V(shape<0>(blockB) == shape<1>(blockC));      // BLK_FW
    CUTE_STATIC_ASSERT_V(shape<1>(blockA) == shape<1>(blockB)); // BLK_K

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ABlockLayout>];
    __shared__ TB smemB[cosize_v<BBlockLayout>];
    auto sA = make_tensor(make_smem_ptr(smemA), blockA); // (BLK_M,BLK_K)
    auto sB = make_tensor(make_smem_ptr(smemB), blockB); // (BLK_FW,BLK_K)

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(BH, M, K), dA);     // (BH,M,K)
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(BH, M, FW, K), dB); // (BH,M,FW,K)
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(BH, M, FW), dC);    // (BH,M,FW)

    // Get the appropriate blocks for this thread block --
    // potential for thread block locality
    auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB)); // (BLK_M,BLK_FW,BLK_K)
    auto blk_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.z, _); // (bh,m,w,k)

    auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, _1, X, _1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, blk_shape, blk_coord, Step<_1, _1, _1, _1>{}); // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1, _1, _1, X>{});  // (BLK_M,BLK_N)

    // Partition the copying of A and B tiles across the threads
    auto tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
    auto tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)

    auto tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
    auto tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)

    // Define C accumulators and A/B partitioning
    // Partition sA (M,K) by the rows of tC
    auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    auto tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC); // (THR_M,THR_N)

    // Clear the accumulators
    clear(tCrC);

    auto k_max = size<3>(tAgA);

    for (int k = 0; k < k_max; ++k)
    {
        // Copy gmem to smem
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);

        cp_async_fence();
        cp_async_wait<0>();

        __syncthreads();

        // Compute gemm on smem
        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }
}

template <class BHShape, class MShape, class KShape, class FWShape,
          class TA, class AStride, class ABlockLayout, class AThreadLayout,
          class TB, class BStride, class BBlockLayout, class BThreadLayout,
          class TC, class CStride, class CBlockLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void gbmm_nn_bn_device(BHShape BH, MShape M, KShape K, FWShape FW,
                                                                                                   TA const *A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
                                                                                                   TB const *B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
                                                                                                   TC *C, CStride dC, CBlockLayout, CThreadLayout tC)
{
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

    CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
    CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

    // CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));      // BLK_M
    // CUTE_STATIC_ASSERT_V(shape<1>(blockB) == shape<1>(blockC));      // BLK_K
    CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockB)); // BLK_M

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ABlockLayout>];
    __shared__ TB smemB[cosize_v<BBlockLayout>];
    auto sA = make_tensor(make_smem_ptr(smemA), blockA); // (BLK_M,BLK_FW)
    auto sB = make_tensor(make_smem_ptr(smemB), blockB); // (BLK_M,BLK_K)

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(BH, M, FW), dA); // (BH,M,FW)
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(BH, M, K), dB);  // (BH,M,K)
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(BH, M, K), dC);  // (BH,M,K)

    // Get the appropriate blocks for this thread block --
    // potential for thread block locality
    auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB)); // (BLK_M,BLK_M,BLK_K)
    auto blk_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.z, _); // (bh,m,w,k)

    auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, _1, X, _1>{});  // (BLK_M,BLK_K,k)
    auto gB = local_tile(mB, blk_shape, blk_coord, Step<_1, _1, _1, _1>{}); // (BLK_N,BLK_K,k)
    auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1, _1, _1, X>{});  // (BLK_M,BLK_N)

    // Partition the copying of A and B tiles across the threads
    auto tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
    auto tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)

    auto tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
    auto tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)

    // Define C accumulators and A/B partitioning
    // Partition sA (M,K) by the rows of tC
    auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    auto tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC); // (THR_M,THR_N)

    // Clear the accumulators
    clear(tCrC);

    auto k_max = size<3>(tAgA);

    for (int k = 0; k < k_max; ++k)
    {
        // Copy gmem to smem
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);

        cp_async_fence();
        cp_async_wait<0>();

        __syncthreads();

        // Compute gemm on smem
        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }
}

template <typename TA, typename TB, typename TC, int bM, int bK, int bW, int W>
void gbmm_nt_nn(int bh, int m, int k,
                TA const *A, int ldA,
                TB const *B, int ldB,
                TC *C, int ldC,
                cudaStream_t stream = 0)
{
    // both A and B are BH x M x K non-banded matrices
    // C is the BH x M x 2W+1 banded matrix of the multiplication between A and B
    using namespace cute;

    auto BH = int(bh);
    auto M = int(m);
    auto K = int(k);
    auto FW = 2 * W + 1; // full window
    auto MP = M + W * 2; // padded M

    // Define strides (mixed)
    auto dA = make_stride(M * K, K, Int<1>{});
    auto dB = make_stride(MP * K, K, K, Int<1>{});
    auto dC = make_stride(M * FW, FW, Int<1>{});

    // Define the block layouts (static)
    auto sA = make_layout(make_shape(bM, bK));
    auto sB = make_layout(make_shape(bW, bK));
    auto sC = make_layout(make_shape(bW, bK));

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    dim3 dimBlock(size(tC));
    dim3 dimGrid(BH,
                 ceil_div(size(M), size(bM)),
                 ceil_div(size(FW), size(bW)));
    gbmm_nt_nn_device<<<dimGrid, dimBlock, 0, stream>>>(BH, M, K, FW,
                                                        A, dA, sA, tA,
                                                        B, dB, sB, tB,
                                                        C, dC, sC, tC);
}

template <typename TA, typename TB, typename TC, int bM, int bK, int bW, int W>
void gbmm_nn_bn(int bh, int m, int k,
                TA const *A, int ldA,
                TB const *B, int ldB,
                TC *C, int ldC,
                cudaStream_t stream = 0)
{
    // A is a banded BH x M x 2W+1 matrix, B is a non-banded BH x M x K matrix
    // C is the BH x M x K non-banded matrix of the multiplication between A and B
    using namespace cute;

    auto BH = int(bh);
    auto M = int(m);
    auto K = int(k);
    auto FW = 2 * W + 1; // full window

    // Define strides (mixed)
    auto dA = make_stride(M * FW, FW, Int<1>{});
    auto dB = make_stride(K * M, K, Int<1>{});
    auto dC = make_stride(K * M, K, Int<1>{});

    // Define the block layouts (static)
    auto sA = make_layout(make_shape(bM, bW));
    auto sB = make_layout(make_shape(bM, bK));
    auto sC = make_layout(make_shape(bM, bK));

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    dim3 dimBlock(size(tC));
    dim3 dimGrid(BH,
                 ceil_div(size(M), size(bM)),
                 ceil_div(size(FW), size(bW)));
    gbmm_nn_bn_device<<<dimGrid, dimBlock, 0, stream>>>(BH, M, K, FW,
                                                        A, dA, sA, tA,
                                                        B, dB, sB, tB,
                                                        C, dC, sC, tC);
}