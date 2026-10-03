# Reproducibility container for ZASSD (RTX 4050 6GB reference testbed).
# Build: docker build -t zassd:cu128 .
# CPU-only checks (no GPU needed):
#   docker run --rm zassd:cu128 pytest tests/test_roofline.py tests/test_hybrid_router.py -q
#   docker run --rm -v $PWD/experiments:/w/experiments zassd:cu128 python3 scripts/run_roofline_analysis.py
# Full GPU repro (needs NVIDIA Container Toolkit + cached HF models):
#   docker run --rm --gpus all -v $PWD/experiments:/w/experiments zassd:cu128 \
#     python3 scripts/run_standard_benchmark_suite.py --samples-per-bench 10
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /w
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0 PYTHONPATH=/w/src

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install -e . && pip install scipy

COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY data/benchmarks/ ./data/benchmarks/

CMD ["pytest", "tests/test_roofline.py", "tests/test_hybrid_router.py", "-q"]
