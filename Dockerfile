# docker build -t fschlatt/window-matmul:0.0.1 .
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y build-essential g++ git
COPY . /window_matmul
RUN cd /window_matmul && pip install .

FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

COPY --from=0 /opt/conda/lib/python3.10/site-packages/window_matmul /opt/conda/lib/python3.10/site-packages/window_matmul
COPY --from=0 /opt/conda/lib/python3.10/site-packages/window_matmul-0.1.0.dist-info /opt/conda/lib/python3.10/site-packages/window_matmul-0.1.0.dist-info
COPY --from=0 /opt/conda/lib/python3.10/site-packages/window_matmul_kernel.so /opt/conda/lib/python3.10/site-packages/window_matmul_kernel.so
