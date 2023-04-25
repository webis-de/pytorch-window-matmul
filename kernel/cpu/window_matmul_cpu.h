#pragma once

#include <torch/torch.h>

torch::Tensor window_matmul_fw_cpu(torch::Tensor src, torch::Tensor other, int window_size);
std::tuple<torch::Tensor, torch::Tensor> window_matmul_bw_cpu(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output);
torch::Tensor unwindow_matmul_fw_cpu(torch::Tensor src, torch::Tensor other, int window_size);
std::tuple<torch::Tensor, torch::Tensor> unwindow_matmul_bw_cpu(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output);
