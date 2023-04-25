#include "window_matmul_cpu.h"

#include "utils.h"
#include <algorithm>

torch::Tensor window_matmul_fw_cpu(torch::Tensor src, torch::Tensor other, int window_size)
{
  CHECK_CPU(src);
  CHECK_CPU(other);

  CHECK_INPUT(src.dim() == other.dim());
  CHECK_INPUT(src.size(0) == src.size(0));
  CHECK_INPUT(src.size(-1) == other.size(-2));
  CHECK_INPUT(src.size(-2) == other.size(-1));

  src = src.contiguous();
  other = other.contiguous();

  torch::Tensor out;
  auto sizes = src.sizes().vec();
  sizes[2] = window_size * 2 + 1;
  out = torch::zeros(sizes, src.options());
  int batch_size, seq_len, hidden_dim, w_idx;
  batch_size = src.size(0);
  seq_len = src.size(1);
  hidden_dim = src.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out.scalar_type(), "window_matmul_fw_cpu", [&]
      {
        auto src_accessor = src.accessor<scalar_t, 3>();
        auto other_accessor = other.accessor<scalar_t, 3>();
        auto out_accessor = out.accessor<scalar_t, 3>();

        for (auto b = 0; b < batch_size; b++)
        {
          for (auto s = 0; s < seq_len; s++)
          {
            for (auto w = std::max(0, s - window_size); w < std::min(seq_len, s + window_size + 1); w++)
            {
              w_idx = w - std::min(0, s - window_size) - std::max(0, s - window_size);
              for (auto h = 0; h < hidden_dim; h++) {
                out_accessor[b][s][w_idx] += src_accessor[b][s][h] * other_accessor[b][h][w];
              }
            }
          }
        } });

  return out;
}

std::tuple<torch::Tensor, torch::Tensor> window_matmul_bw_cpu(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output)
{
  CHECK_CPU(src);
  CHECK_CPU(other);

  CHECK_INPUT(src.dim() == other.dim());
  CHECK_INPUT(src.size(0) == src.size(0));
  CHECK_INPUT(src.size(-1) == other.size(-2));
  CHECK_INPUT(src.size(-2) == other.size(-1));

  src = src.contiguous();
  other = other.contiguous();

  torch::Tensor grad_src, grad_other;
  grad_src = torch::zeros(src.sizes().vec(), src.options());
  grad_other = torch::zeros(other.sizes().vec(), other.options());
  int batch_size, seq_len, hidden_dim, w_idx;
  batch_size = src.size(0);
  seq_len = src.size(1);
  hidden_dim = src.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_src.scalar_type(), "window_matmul_bw_cpu", [&]
      {
        auto src_accessor = src.accessor<scalar_t, 3>();
        auto other_accessor = other.accessor<scalar_t, 3>();
        auto grad_output_accessor = grad_output.accessor<scalar_t, 3>();
        auto grad_src_accessor = grad_src.accessor<scalar_t, 3>();
        auto grad_other_accessor = grad_other.accessor<scalar_t, 3>();

        for (auto b = 0; b < batch_size; b++)
        {
          for (auto s = 0; s < seq_len; s++)
          {
            for (auto w = std::max(0, s - window_size); w < std::min(seq_len, s + window_size + 1); w++)
            {
              w_idx = w - std::min(0, s - window_size) - std::max(0, s - window_size);
              for (auto h = 0; h < hidden_dim; h++)
              {
                grad_src_accessor[b][s][h] += other_accessor[b][h][w] * grad_output_accessor[b][s][w_idx];
                grad_other_accessor[b][h][s] += src_accessor[b][w][h] * grad_output_accessor[b][w][2 * window_size - w_idx];
              }
            }
          }
        } });
  return std::make_tuple(grad_src, grad_other);
}

torch::Tensor unwindow_matmul_fw_cpu(torch::Tensor src, torch::Tensor other, int window_size)
{
  CHECK_CPU(src);
  CHECK_CPU(other);

  CHECK_INPUT(src.dim() == other.dim());
  CHECK_INPUT(src.size(0) == src.size(0));
  CHECK_INPUT(src.size(-1) == window_size * 2 + 1);
  CHECK_INPUT(src.size(-2) == other.size(-2));

  src = src.contiguous();
  other = other.contiguous();

  torch::Tensor out;
  auto sizes = other.sizes().vec();
  out = torch::zeros(sizes, src.options());
  int batch_size, seq_len, hidden_dim, w_idx;
  batch_size = other.size(0);
  seq_len = other.size(1);
  hidden_dim = other.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out.scalar_type(), "unwindow_matmul_fw_cpu", [&]
      {
        auto src_accessor = src.accessor<scalar_t, 3>();
        auto other_accessor = other.accessor<scalar_t, 3>();
        auto out_accessor = out.accessor<scalar_t, 3>();

        for (auto b = 0; b < batch_size; b++)
        {
          for (auto s = 0; s < seq_len; s++)
          {
            for (auto w = std::max(0, s - window_size); w < std::min(seq_len, s + window_size + 1); w++)
            {
              w_idx = w - std::min(0, s - window_size) - std::max(0, s - window_size);
              for (auto h = 0; h < hidden_dim; h++) {
                out_accessor[b][s][h] += src_accessor[b][s][w_idx] * other_accessor[b][w][h];
              }
            }
          }
        } });

  return out;
}

std::tuple<torch::Tensor, torch::Tensor> unwindow_matmul_bw_cpu(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output)
{
  CHECK_CPU(src);
  CHECK_CPU(other);

  CHECK_INPUT(src.dim() == other.dim());
  CHECK_INPUT(src.size(0) == src.size(0));
  CHECK_INPUT(src.size(-1) == window_size * 2 + 1);
  CHECK_INPUT(src.size(-2) == other.size(-2));

  src = src.contiguous();
  other = other.contiguous();

  torch::Tensor grad_src, grad_other;
  grad_src = torch::zeros(src.sizes().vec(), src.options());
  grad_other = torch::zeros(other.sizes().vec(), other.options());
  int batch_size, seq_len, hidden_dim, w_idx;
  batch_size = other.size(0);
  seq_len = other.size(1);
  hidden_dim = other.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_src.scalar_type(), "unwindow_matmul_bw_cpu", [&]
      {
        auto src_accessor = src.accessor<scalar_t, 3>();
        auto other_accessor = other.accessor<scalar_t, 3>();
        auto grad_output_accessor = grad_output.accessor<scalar_t, 3>();
        auto grad_src_accessor = grad_src.accessor<scalar_t, 3>();
        auto grad_other_accessor = grad_other.accessor<scalar_t, 3>();

        for (auto b = 0; b < batch_size; b++)
        {
          for (auto s = 0; s < seq_len; s++)
          {
            for (auto w = std::max(0, s - window_size); w < std::min(seq_len, s + window_size + 1); w++)
            {
              w_idx = w - std::min(0, s - window_size) - std::max(0, s - window_size);
              for (auto h = 0; h < hidden_dim; h++)
              {
                grad_src_accessor[b][s][w_idx] += other_accessor[b][w][h] * grad_output_accessor[b][s][h];
                grad_other_accessor[b][s][h] += src_accessor[b][w][2 * window_size - w_idx] * grad_output_accessor[b][w][h];
              }
            }
          }
        } });
  return std::make_tuple(grad_src, grad_other);
}