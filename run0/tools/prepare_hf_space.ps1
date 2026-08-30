[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Destination)

$ErrorActionPreference = "Stop"
$Run0Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DestinationFull = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $DestinationFull) {
    throw "Destination already exists; refusing to merge or overwrite: $DestinationFull"
}
New-Item -ItemType Directory -Path $DestinationFull | Out-Null

foreach ($Name in @("README.md", "Dockerfile", "requirements.txt", "worker.py", "generate_open_audio.py", "job_requirements.txt", "review_wheels.py", "job_entrypoint.sh", "train_requirements.txt", "train_job_entrypoint.sh", "run_training_job.py", ".dockerignore")) {
    Copy-Item -LiteralPath (Join-Path $Run0Root "hf_space\$Name") -Destination $DestinationFull
}
foreach ($Name in @("train.py", "script.py", "generate_audio.py", "config.toml", "corpus_sources.json")) {
    Copy-Item -LiteralPath (Join-Path $Run0Root $Name) -Destination $DestinationFull
}
$PackageDestination = New-Item -ItemType Directory -Path (Join-Path $DestinationFull "deepvoice_run0")
Get-ChildItem -LiteralPath (Join-Path $Run0Root "deepvoice_run0") -File |
    Where-Object { $_.Extension -eq ".py" } |
    Copy-Item -Destination $PackageDestination.FullName
$PromptDestination = New-Item -ItemType Directory -Path (Join-Path $DestinationFull "prompts")
Get-ChildItem -LiteralPath (Join-Path $Run0Root "prompts") -File |
    Copy-Item -Destination $PromptDestination.FullName
Copy-Item -LiteralPath (Join-Path $Run0Root "tools\manifest_cli.py") -Destination (Join-Path $DestinationFull "manifest_cli.py")
Copy-Item -LiteralPath (Join-Path $Run0Root "backbone_source.json") -Destination $DestinationFull
Write-Output "Prepared standalone Docker Space source at $DestinationFull"
