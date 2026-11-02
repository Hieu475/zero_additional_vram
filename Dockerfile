# Container for the B1 exact quantized speculative-decoding study.
# Build: docker build -t zassd-b1 .
# Unit tests (CPU):  docker run --rm zassd-b1
# GPU pilot (needs NVIDIA Container Toolkit and cached HF models):
#   docker run --rm --gpus all -v $PWD/experiments:/w/experiments zassd-b1 bash reproduce.sh
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /w
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0 PYTHONPATH=/w/src:/w/scripts

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install -e . && pip install scipy

COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/benchmarks/ ./data/benchmarks/
COPY reproduce.sh ./

CMD ["python3", "-m", "pytest", "tests", "-q"]
