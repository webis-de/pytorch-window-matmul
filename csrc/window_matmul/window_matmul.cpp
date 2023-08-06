#include <torch/extension.h>
#include "cpu/window_matmul_cpu.h"

#ifdef WITH_CUDA
#include "cuda/window_matmul_cuda.h"
#endif

torch::Tensor window_matmul_fw(torch::Tensor src, torch::Tensor other, int window_size)
{
  if (src.device().is_cuda())
  {
#ifdef WITH_CUDA
    return window_matmul_fw_cuda(src, other, window_size);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
  else
  {
    return window_matmul_fw_cpu(src, other, window_size);
  }
}

std::tuple<torch::Tensor, torch::Tensor> window_matmul_bw(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output)
{
  if (src.device().is_cuda())
  {
#ifdef WITH_CUDA
    return window_matmul_bw_cuda(src, other, window_size, grad_output);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
  else
  {
    return window_matmul_bw_cpu(src, other, window_size, grad_output);
  }
}

torch::Tensor unwindow_matmul_fw(torch::Tensor src, torch::Tensor other, int window_size)
{
  if (src.device().is_cuda())
  {
#ifdef WITH_CUDA
    return unwindow_matmul_fw_cuda(src, other, window_size);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
  else
  {
    return unwindow_matmul_fw_cpu(src, other, window_size);
  }
}

std::tuple<torch::Tensor, torch::Tensor> unwindow_matmul_bw(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output)
{
  if (src.device().is_cuda())
  {
#ifdef WITH_CUDA
    return unwindow_matmul_bw_cuda(src, other, window_size, grad_output);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
  else
  {
    return unwindow_matmul_bw_cpu(src, other, window_size, grad_output);
  }
}

static torch::Tensor window_matmul_forward(torch::Tensor src, torch::Tensor other, int window_size)
{
  auto out = window_matmul_fw(src, other, window_size);
  return out;
}

static std::tuple<torch::Tensor, torch::Tensor> window_matmul_backward(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output)
{
  auto out = window_matmul_bw(src, other, window_size, grad_output);
  return out;
}

static torch::Tensor unwindow_matmul_forward(torch::Tensor src, torch::Tensor other, int window_size)
{
  auto out = unwindow_matmul_fw(src, other, window_size);
  return out;
}

static std::tuple<torch::Tensor, torch::Tensor> unwindow_matmul_backward(
    torch::Tensor src, torch::Tensor other, int window_size, torch::Tensor grad_output)
{
  auto out = unwindow_matmul_bw(src, other, window_size, grad_output);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("window_matmul_forward", &window_matmul_forward, "WindowMatmul forward");
  m.def("window_matmul_backward", &window_matmul_backward, "WindowMatmul backward");
  m.def("unwindow_matmul_forward", &unwindow_matmul_forward, "UnwindowMatmul forward");
  m.def("unwindow_matmul_backward", &unwindow_matmul_backward, "UnwindowMatmul backward");
}
