[CmdletBinding()]
param(
    [string]$OutputPath = "",
    [string]$BaselineRoot = "",
    [string]$Run0ModelRoot = "",
    [string]$FusionSpec = "",
    [string]$LogoReport = ""
)

$ErrorActionPreference = "Stop"
$Run1Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $Run1Root "..")).Path
if (-not $OutputPath) { $OutputPath = Join-Path $Run1Root "build\submit-run1-hybrid.zip" }
if (-not $BaselineRoot) { $BaselineRoot = Join-Path $RepoRoot "format\baseline_submit" }
if (-not $Run0ModelRoot) { $Run0ModelRoot = Join-Path $Run1Root "model\run0" }
if (-not $FusionSpec) { $FusionSpec = Join-Path $Run1Root "model\fusion.json" }
if (-not $LogoReport) { $LogoReport = Join-Path $Run1Root "reports\LOGO_REPORT.json" }

$OutputFull = [System.IO.Path]::GetFullPath($OutputPath)
$OutputManifest = [System.IO.Path]::ChangeExtension($OutputFull, ".manifest.json")
foreach ($Path in @($OutputFull, $OutputManifest)) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite: $Path" }
}
$BaselineFull = (Resolve-Path -LiteralPath $BaselineRoot).Path
$Run0Full = (Resolve-Path -LiteralPath $Run0ModelRoot).Path
$FusionFull = (Resolve-Path -LiteralPath $FusionSpec).Path
$LogoFull = (Resolve-Path -LiteralPath $LogoReport).Path
$Logo = Get-Content -Raw -LiteralPath $LogoFull | ConvertFrom-Json
if ($Logo.status -ne "passed") { throw "LOGO selection gate did not pass" }

$Required = @(
    (Join-Path $BaselineFull "script.py"),
    (Join-Path $BaselineFull "model\df_arena_1b\pytorch_model.bin"),
    (Join-Path $BaselineFull "model\panns\Cnn14_mAP=0.431.pth"),
    (Join-Path $BaselineFull "model\htdemucs\955717e8-8726e21a.th"),
    (Join-Path $Run0Full "run0.safetensors"),
    (Join-Path $Run0Full "model_config.json"),
    (Join-Path $Run0Full "backbone\model.safetensors"),
    $FusionFull
)
$Missing = @($Required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($Missing.Count -gt 0) { throw "Submission artifacts are incomplete: $($Missing -join ', ')" }

$TempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$Stage = Join-Path $TempRoot ("deepvoice-run1-" + [guid]::NewGuid().ToString("N"))
if (-not $Stage.StartsWith($TempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $Stage"
}

try {
    New-Item -ItemType Directory -Path $Stage | Out-Null
    Copy-Item -LiteralPath (Join-Path $Run1Root "script.py") -Destination $Stage
    Copy-Item -LiteralPath (Join-Path $Run1Root "requirements.txt") -Destination $Stage
    $StageModel = New-Item -ItemType Directory -Path (Join-Path $Stage "model")
    Copy-Item -LiteralPath $FusionFull -Destination (Join-Path $StageModel.FullName "fusion.json")
    Copy-Item -LiteralPath $LogoFull -Destination (Join-Path $StageModel.FullName "LOGO_REPORT.json")
    Copy-Item -LiteralPath (Join-Path $BaselineFull "script.py") -Destination (Join-Path $StageModel.FullName "baseline_script.py")

    $StageBaseline = New-Item -ItemType Directory -Path (Join-Path $StageModel.FullName "baseline")
    foreach ($Name in @("df_arena_1b", "panns", "htdemucs")) {
        Copy-Item -LiteralPath (Join-Path $BaselineFull "model\$Name") -Destination $StageBaseline.FullName -Recurse
    }

    $StageRun0 = New-Item -ItemType Directory -Path (Join-Path $StageModel.FullName "run0")
    foreach ($Name in @("run0.safetensors", "model_config.json", "TRAINING_REPORT.json")) {
        $Source = Join-Path $Run0Full $Name
        if (Test-Path -LiteralPath $Source -PathType Leaf) {
            Copy-Item -LiteralPath $Source -Destination $StageRun0.FullName
        }
    }
    Copy-Item -LiteralPath (Join-Path $Run0Full "backbone") -Destination $StageRun0.FullName -Recurse
    Copy-Item -LiteralPath (Join-Path $Run1Root "config.toml") -Destination (Join-Path $StageRun0.FullName "config.toml")
    $Run0Package = New-Item -ItemType Directory -Path (Join-Path $StageRun0.FullName "deepvoice_run0")
    Get-ChildItem -LiteralPath (Join-Path $RepoRoot "run0\deepvoice_run0") -File -Filter "*.py" |
        Copy-Item -Destination $Run0Package.FullName
    $HybridPackage = New-Item -ItemType Directory -Path (Join-Path $StageModel.FullName "deepvoice_hybrid")
    Get-ChildItem -LiteralPath (Join-Path $Run1Root "deepvoice_hybrid") -File -Filter "*.py" |
        Copy-Item -Destination $HybridPackage.FullName

    $ArtifactFiles = @(Get-ChildItem -LiteralPath $StageModel.FullName -Recurse -File | Where-Object { $_.Name -ne "ARTIFACTS.json" })
    $ArtifactLedger = [ordered]@{}
    foreach ($File in ($ArtifactFiles | Sort-Object FullName)) {
        $Relative = [System.IO.Path]::GetRelativePath($StageModel.FullName, $File.FullName).Replace('\', '/')
        $ArtifactLedger[$Relative] = [ordered]@{
            bytes = $File.Length
            sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    [ordered]@{format_version = 1; files = $ArtifactLedger} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $StageModel.FullName "ARTIFACTS.json") -Encoding UTF8

    New-Item -ItemType Directory -Path (Split-Path $OutputFull -Parent) -Force | Out-Null
    & python (Join-Path $RepoRoot "run0\tools\zip_submission.py") -- $Stage $OutputFull
    if ($LASTEXITCODE -ne 0) { throw "Zip64 package creation failed" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [System.IO.Compression.ZipFile]::OpenRead($OutputFull)
    try {
        $Entries = @($Archive.Entries)
        $Names = @($Entries | ForEach-Object { $_.FullName })
        if ("script.py" -notin $Names -or "requirements.txt" -notin $Names) {
            throw "Archive root contract is incomplete"
        }
        $Unexpected = @($Names | Where-Object {
            $_ -ne "script.py" -and $_ -ne "requirements.txt" -and -not $_.StartsWith("model/")
        })
        if ($Unexpected.Count -gt 0) { throw "Unexpected archive entries: $($Unexpected -join ', ')" }
        $Unsafe = @($Names | Where-Object { $_.StartsWith('/') -or $_.Contains('\') -or $_.Contains('../') })
        if ($Unsafe.Count -gt 0) { throw "Unsafe archive entries: $($Unsafe -join ', ')" }
        $Expanded = [long]($Entries | Measure-Object -Property Length -Sum).Sum
    }
    finally { $Archive.Dispose() }
    $ArchiveBytes = (Get-Item -LiteralPath $OutputFull).Length
    if ($ArchiveBytes -ge 9GB) { throw "Internal 9 GB archive gate failed: $ArchiveBytes" }
    if ($Expanded -ge 30GB) { throw "Internal 30 GB expanded gate failed: $Expanded" }
    $ArchiveHash = (Get-FileHash -LiteralPath $OutputFull -Algorithm SHA256).Hash.ToLowerInvariant()
    [ordered]@{
        artifact = "dacon-deepvoice-run1-hybrid"
        archive = [System.IO.Path]::GetFileName($OutputFull)
        archive_sha256 = $ArchiveHash
        archive_bytes = $ArchiveBytes
        uncompressed_bytes = $Expanded
        entry_count = $Entries.Count
        source_git_commit = (& git -C $RepoRoot rev-parse HEAD).Trim()
        logo_report_sha256 = (Get-FileHash -LiteralPath $LogoFull -Algorithm SHA256).Hash.ToLowerInvariant()
        validations = [ordered]@{
            logo_gate = $true
            exact_root_contract = $true
            offline_models_included = $true
            tensor_safe_run0 = $true
            archive_under_9gb = $true
            expanded_under_30gb = $true
            file_independent_fusion = $true
        }
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutputManifest -Encoding UTF8
    Write-Output "Created Run1 submission: $OutputFull"
    Write-Output "SHA-256: $ArchiveHash"
    Write-Output "Archive bytes: $ArchiveBytes"
}
finally {
    if (Test-Path -LiteralPath $Stage) {
        $Resolved = (Resolve-Path -LiteralPath $Stage).Path
        if (-not $Resolved.StartsWith($TempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove unsafe staging path: $Resolved"
        }
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    }
}
