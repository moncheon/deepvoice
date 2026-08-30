[CmdletBinding()]
param([string]$OutputPath = "")

$ErrorActionPreference = "Stop"
$Run0Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputPath) {
    $OutputPath = Join-Path $Run0Root "build\submit-run0-v1.zip"
}
$OutputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$ManifestFullPath = [System.IO.Path]::ChangeExtension($OutputFullPath, ".manifest.json")
foreach ($Path in @($OutputFullPath, $ManifestFullPath)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite existing submission artifact: $Path"
    }
}

$ModelRoot = Join-Path $Run0Root "model"
$RequiredArtifacts = @(
    (Join-Path $ModelRoot "run0.safetensors"),
    (Join-Path $ModelRoot "model_config.json"),
    (Join-Path $ModelRoot "TRAINING_REPORT.json"),
    (Join-Path $ModelRoot "backbone\config.json"),
    (Join-Path $ModelRoot "backbone\model.safetensors"),
    (Join-Path $ModelRoot "backbone\LICENSE-CC-BY-SA-3.0.txt"),
    (Join-Path $ModelRoot "backbone\ORIGIN.json")
)
$Missing = @($RequiredArtifacts | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($Missing.Count -gt 0) {
    throw "Model artifact contract is incomplete: $($Missing -join ', ')"
}

$Metadata = Get-Content -Raw -LiteralPath (Join-Path $ModelRoot "model_config.json") | ConvertFrom-Json
if ($Metadata.format_version -ne 2) {
    throw "model_config.json must use format_version 2"
}
if ($Metadata.training.fixture_smoke_training -ne $false) {
    throw "Fixture-trained weights cannot be packaged for performance submission"
}
if ([int]$Metadata.training.epochs_completed -lt 4) {
    throw "At least four training epochs are required"
}
$DevScore = [double]$Metadata.best_dev_metrics.weighted_score
$HoldoutScore = [double]$Metadata.holdout_metrics.weighted_score
if ($HoldoutScore -le 0.55) {
    throw "Holdout weighted score gate failed: $HoldoutScore"
}
if (($DevScore - $HoldoutScore) -gt 0.15) {
    throw "Dev-holdout score gap exceeds 0.15"
}
foreach ($Name in @(
    "std_file_fake_prob", "std_voice_fake_prob", "std_music_fake_prob",
    "std_voice_present_prob", "std_music_present_prob"
)) {
    if ([double]$Metadata.holdout_metrics.$Name -le 0.01) {
        throw "Holdout prediction variance gate failed: $Name"
    }
}

$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$StageFull = Join-Path $TempBase ("deepvoice-run0-submit-" + [guid]::NewGuid().ToString("N"))
if (-not $StageFull.StartsWith($TempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $StageFull"
}

try {
    New-Item -ItemType Directory -Path $StageFull | Out-Null
    Copy-Item -LiteralPath (Join-Path $Run0Root "script.py") -Destination $StageFull
    Copy-Item -LiteralPath (Join-Path $Run0Root "requirements.txt") -Destination $StageFull
    $StageModel = New-Item -ItemType Directory -Path (Join-Path $StageFull "model")
    foreach ($Name in @("run0.safetensors", "model_config.json", "TRAINING_REPORT.json")) {
        Copy-Item -LiteralPath (Join-Path $ModelRoot $Name) -Destination $StageModel.FullName
    }
    if (Test-Path -LiteralPath (Join-Path $ModelRoot "fuser.json")) {
        Copy-Item -LiteralPath (Join-Path $ModelRoot "fuser.json") -Destination $StageModel.FullName
    }
    Copy-Item -LiteralPath (Join-Path $ModelRoot "backbone") -Destination $StageModel.FullName -Recurse
    Copy-Item -LiteralPath (Join-Path $Run0Root "config.toml") -Destination $StageModel.FullName
    $StagePackage = New-Item -ItemType Directory -Path (Join-Path $StageModel.FullName "deepvoice_run0")
    Get-ChildItem -LiteralPath (Join-Path $Run0Root "deepvoice_run0") -File -Filter "*.py" |
        Copy-Item -Destination $StagePackage.FullName

    $StagedFiles = @(Get-ChildItem -LiteralPath $StageFull -Recurse -File)
    if ($StagedFiles.Count -eq 0) {
        throw "Submission staging unexpectedly contains no files"
    }

    New-Item -ItemType Directory -Path (Split-Path $OutputFullPath -Parent) -Force | Out-Null
    & python (Join-Path $PSScriptRoot "zip_submission.py") -- $StageFull $OutputFullPath
    if ($LASTEXITCODE -ne 0) {
        throw "Streaming Zip64 package creation failed with exit code $LASTEXITCODE"
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [System.IO.Compression.ZipFile]::OpenRead($OutputFullPath)
    try {
        $Entries = @($Archive.Entries)
        $Names = @($Entries | ForEach-Object { $_.FullName })
        foreach ($Required in @("script.py", "requirements.txt")) {
            if ($Required -notin $Names) {
                throw "Archive is missing required root file: $Required"
            }
        }
        if (-not ($Names | Where-Object { $_.StartsWith("model/") })) {
            throw "Archive is missing model/ content"
        }
        $Unexpected = @($Names | Where-Object {
            $_ -ne "script.py" -and $_ -ne "requirements.txt" -and -not $_.StartsWith("model/")
        })
        if ($Unexpected.Count -gt 0) {
            throw "Archive contains unexpected top-level content: $($Unexpected -join ', ')"
        }
        $Unsafe = @($Names | Where-Object {
            $_.StartsWith('/') -or $_.Contains('\\') -or $_.Contains('../')
        })
        if ($Unsafe.Count -gt 0) {
            throw "Archive contains unsafe paths: $($Unsafe -join ', ')"
        }
        $ExpandedBytes = [long]($Entries | Measure-Object -Property Length -Sum).Sum
    }
    finally {
        $Archive.Dispose()
    }
    $ArchiveBytes = (Get-Item -LiteralPath $OutputFullPath).Length
    if ($ArchiveBytes -ge 10GB) {
        throw "Archive exceeds the official 10 GB limit: $ArchiveBytes"
    }
    if ($ExpandedBytes -ge 32GB) {
        throw "Archive exceeds the official 32 GB expanded limit: $ExpandedBytes"
    }
    $ArchiveHash = (Get-FileHash -LiteralPath $OutputFullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $Manifest = [ordered]@{
        artifact = "dacon-deepvoice-run0-v1"
        archive = [System.IO.Path]::GetFileName($OutputFullPath)
        archive_sha256 = $ArchiveHash
        archive_bytes = $ArchiveBytes
        uncompressed_bytes = $ExpandedBytes
        entry_count = $Entries.Count
        source_git_commit = (& git -C $Run0Root rev-parse HEAD).Trim()
        best_dev_weighted_score = $DevScore
        holdout_weighted_score = $HoldoutScore
        training_run_id = $Metadata.provenance.run_id
        validations = [ordered]@{
            performance_gate = $true
            exact_root_contract = $true
            offline_backbone_included = $true
            tensor_safe_weights = $true
            archive_under_10gb = $true
            expanded_under_32gb = $true
        }
    }
    $Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestFullPath -Encoding UTF8
    Write-Output "Created Run0 submission: $OutputFullPath"
    Write-Output "SHA-256: $ArchiveHash"
    Write-Output "Archive bytes: $ArchiveBytes"
    Write-Output "Expanded bytes: $ExpandedBytes"
    Write-Output "Manifest: $ManifestFullPath"
}
finally {
    if (Test-Path -LiteralPath $StageFull) {
        $ResolvedStage = (Resolve-Path -LiteralPath $StageFull).Path
        if (-not $ResolvedStage.StartsWith($TempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove unsafe staging path: $ResolvedStage"
        }
        Remove-Item -LiteralPath $ResolvedStage -Recurse -Force
    }
}
