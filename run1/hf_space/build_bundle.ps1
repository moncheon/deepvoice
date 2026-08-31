[CmdletBinding()]
param([string]$OutputPath = "")

$ErrorActionPreference = "Stop"
$Run1Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $Run1Root "..")).Path
if (-not $OutputPath) { $OutputPath = Join-Path $Run1Root "work\hf_bundle" }
$OutputFull = [System.IO.Path]::GetFullPath($OutputPath)
if (-not $OutputFull.StartsWith($Run1Root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Bundle output must stay under run1: $OutputFull"
}
if (Test-Path -LiteralPath $OutputFull) {
    throw "Refusing to overwrite existing bundle: $OutputFull"
}
New-Item -ItemType Directory -Path $OutputFull | Out-Null
foreach ($Name in @(
    "train_requirements.txt", "review_wheels.py", "train_job_entrypoint.sh",
    "final_job_entrypoint.sh",
    "run_logo_job.py", "run_final_job.py", "baseline_requirements.txt", "baseline_artifacts.json",
    "baseline_job_entrypoint.sh", "baseline_dependency_smoke.py",
    "run_baseline_cache_job.py", "benchmark_job_entrypoint.sh", "run_benchmark_job.py"
)) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $Name) -Destination $OutputFull
}
Copy-Item -LiteralPath (Join-Path $Run1Root "config.toml") -Destination $OutputFull
$Run0Package = New-Item -ItemType Directory -Path (Join-Path $OutputFull "deepvoice_run0")
Get-ChildItem -LiteralPath (Join-Path $RepoRoot "run0\deepvoice_run0") -File -Filter "*.py" |
    Copy-Item -Destination $Run0Package.FullName
$HybridPackage = New-Item -ItemType Directory -Path (Join-Path $OutputFull "deepvoice_hybrid")
Get-ChildItem -LiteralPath (Join-Path $Run1Root "deepvoice_hybrid") -File -Filter "*.py" |
    Copy-Item -Destination $HybridPackage.FullName
$BaselineSource = Join-Path $RepoRoot "format\baseline_submit"
Copy-Item -LiteralPath (Join-Path $BaselineSource "script.py") -Destination (Join-Path $OutputFull "baseline_script.py")
$BaselineModel = New-Item -ItemType Directory -Path (Join-Path $OutputFull "baseline_model")
foreach ($Relative in @(
    "df_arena_1b\__init__.py", "df_arena_1b\backbone.py", "df_arena_1b\config.json",
    "df_arena_1b\configuration_antispoofing.py", "df_arena_1b\conformer.py",
    "df_arena_1b\feature_extraction_antispoofing.py", "df_arena_1b\LICENSE.txt",
    "df_arena_1b\modeling_antispoofing.py", "df_arena_1b\pipeline_antispoofing.py",
    "df_arena_1b\preprocessor_config.json", "df_arena_1b\README.md",
    "df_arena_1b\facebook\wav2vec2-xls-r-1b\config.json",
    "htdemucs\htdemucs.yaml", "panns\class_labels_indices.csv",
    "panns\component_labels.json"
)) {
    $Source = Join-Path (Join-Path $BaselineSource "model") $Relative
    $Destination = Join-Path $BaselineModel.FullName $Relative
    New-Item -ItemType Directory -Path (Split-Path $Destination -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination
}
$Files = Get-ChildItem -LiteralPath $OutputFull -File -Recurse | Sort-Object FullName
$Hashes = [ordered]@{}
foreach ($File in $Files) {
    $Relative = [System.IO.Path]::GetRelativePath($OutputFull, $File.FullName).Replace("\", "/")
    $Hashes[$Relative] = [ordered]@{
        bytes = $File.Length
        sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$Manifest = [ordered]@{
    format_version = 1
    files = $Hashes
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputFull "SOURCE_MANIFEST.json") -Encoding utf8NoBOM
$ManifestHash = (Get-FileHash -LiteralPath (Join-Path $OutputFull "SOURCE_MANIFEST.json") -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Output "Created reviewed Run1 HF bundle: $OutputFull"
Write-Output "RUN1_SOURCE_BUNDLE_SHA256=$ManifestHash"
