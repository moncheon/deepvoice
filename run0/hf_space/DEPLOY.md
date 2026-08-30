# Private Docker Space deployment contract

This is a runbook, not an executed deployment. Use a new Space; do not mutate an existing demo.

## 1. Prepare and review

```powershell
powershell -ExecutionPolicy Bypass -File run0/tools/prepare_hf_space.ps1 `
  -Destination D:\temp\deepvoice-run0-space
```

Review every staged file. Do not stage datasets, checkpoints, tokens, local absolute paths, or generated audio.

## 2. Create CPU-only private Space

Authenticate interactively if needed, then substitute your own namespace and unique name:

```powershell
hf auth whoami
hf repos create NAMESPACE/deepvoice-run0-worker --type space --space-sdk docker --private
hf upload NAMESPACE/deepvoice-run0-worker D:\temp\deepvoice-run0-space . --repo-type space
hf spaces settings NAMESPACE/deepvoice-run0-worker --sleep-time 300
```

Set `RUN0_JOB_TOKEN` and any gated-repository `HF_TOKEN` with `hf spaces secrets add -s NAME`, never as public environment variables. Set non-secret job paths with `hf spaces variables add -e NAME=value`. `RUN0_LICENSES_REVIEWED=true` is permitted only after manifest source/license review.

External data and the backbone have a separate staging/review phase. Download a pinned revision without invoking a loader (for example, `hf download REPO --repo-type model --revision REVISION --local-dir /data/staging/backbone`), inspect its card/license/file list, record SHA-256 values, and only then copy the reviewed directory to `/data/output/model/backbone`. The worker never combines download and execution in one job.

## 3. Paid hardware gate

Immediately before attachment, run `hf spaces hardware` and confirm the current price and billing state. The pre-plan budget ceiling is $20; crossing it requires fresh approval. Attach hardware only for a bounded stage, for example:

```powershell
hf spaces settings NAMESPACE/deepvoice-run0-worker --hardware l4x1
hf spaces info NAMESPACE/deepvoice-run0-worker --expand runtime
hf spaces logs NAMESPACE/deepvoice-run0-worker --build --tail 200
hf spaces logs NAMESPACE/deepvoice-run0-worker --tail 200
```

The README frontmatter intentionally does not declare hardware because Space hardware is controlled by settings, not YAML.

## 4. Trigger and verify

Call `GET /health` and `GET /status`. Trigger only an enumerated stage with `POST /run`. For a private Space, `Authorization: Bearer <HF_TOKEN>` authenticates the Hub proxy and `X-Run0-Token: <RUN0_JOB_TOKEN>` authenticates the worker. `build-corpus`, model generation, and training require the reviewed-license gate. Inspect the status, complete runtime logs, output hashes, and model files. `RUNNING` or HTTP 200 alone is not evidence of a valid artifact.

After a successful run, upload `/data/output/model` to an already-created private model repo using `hf upload NAMESPACE/PRIVATE_MODEL_REPO /data/output/model . --repo-type model`, then confirm the remote revision and hashes. This promotion is deliberately outside the worker so a failed or unreviewed run cannot publish itself.

## 5. Stop billing

After confirming the private output repository revision:

```powershell
hf spaces pause NAMESPACE/deepvoice-run0-worker
hf spaces info NAMESPACE/deepvoice-run0-worker --expand runtime
```

No paid Bucket is created by this scaffold. `/data` can be wiped on restart, so it is never the only copy of a checkpoint or ledger.

## 6. Bounded Jobs fallback

If a fine-grained token can write repositories but receives 401 from the Space
hardware endpoint, do not weaken token permissions merely to keep the worker
running. Use an ephemeral Job with a pinned base-image digest and the private
Space repository mounted read-only:

```powershell
hf jobs run --detach `
  --name deepvoice-run0-generate-v1 `
  --timeout 4h `
  --flavor l40sx1 `
  --secrets HF_TOKEN `
  -e RUN0_GENERATED_REPO=NAMESPACE/PRIVATE_CORPUS_REPO `
  -e RUN0_MMS_COUNT=1000 `
  -e RUN0_MUSICGEN_COUNT=256 `
  -e RUN0_AUDIOLDM_COUNT=64 `
  -v hf://spaces/NAMESPACE/deepvoice-run0-worker:/workspace:ro `
  pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755 `
  /bin/bash /workspace/job_entrypoint.sh
```

`job_entrypoint.sh` downloads exact-version wheels without dependencies,
records archive structure/native members and SHA-256 in `WHEEL_REVIEW.json`,
then installs only the reviewed wheelhouse. Pinned model snapshots are likewise
downloaded and recorded in `MODEL_REVIEW.json` before import. A 1+1+1 model
smoke must pass before the bounded production batch. The Job uploads only to an
already-created private dataset repository and terminates automatically on
success, failure, or timeout.

For the production detector run, keep the original generator-holdout split as a
diagnostic and select the reviewed domain-covered ledger explicitly:

```powershell
-e RUN0_SPLIT_RELATIVE=splits/domain-balanced-v1.json
```

The same variable must be passed to the L4 benchmark Job so its 100-file
quality/runtime sample is drawn from the evaluated holdout rather than from a
different ledger. Both training and benchmark reports record the relative path
and SHA-256 of the selected split.

The benchmark copies the promoted safetensors from the read-only Hub mount to
the Job's local `/tmp` filesystem and verifies every copied file against
`ARTIFACTS.json` before model loading. This avoids long-lived mmap access to a
remote FUSE mount. After each 16/12/8-segment pass it uploads
`reports/L4_BENCHMARK_PROGRESS.json`; a late Job failure therefore preserves
the completed measurements even though the final consolidated report is not
yet valid.
