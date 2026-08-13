FROM python:3.11-slim

WORKDIR /app
COPY . .

# ps-general (default): dev extras (tests, docs, mkdocs, multimodal).
# ps-neuroscanner: --build-arg EXTRAS=nnsight,nnterp,vlm pulls in the
# activation-collection backends (torch-heavy, CPU wheels by default).
ARG EXTRAS=dev
RUN pip install --no-cache-dir ".[${EXTRAS}]"

ENTRYPOINT ["psychscanner"]
