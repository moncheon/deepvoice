FROM pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY job_requirements.txt review_wheels.py /workspace/
RUN python -m pip download --only-binary=:all: --no-deps \
      --dest /tmp/run0-wheelhouse --requirement job_requirements.txt && \
    python review_wheels.py /tmp/run0-wheelhouse /workspace/WHEEL_REVIEW.json && \
    python -m pip install --no-cache-dir --no-index --no-deps \
      --find-links /tmp/run0-wheelhouse --requirement job_requirements.txt && \
    rm -r /tmp/run0-wheelhouse
COPY . /workspace

EXPOSE 7860
CMD ["python", "worker.py"]
