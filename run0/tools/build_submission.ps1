[CmdletBinding()]
param(
    [string]$OutputPath = "",
    [switch]$AllowIncompleteModel
)

$ErrorActionPreference = "Stop"
$Run0Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputPath) {
    $OutputPath = Join-Path $Run0Root "build\run0_submission.zip"
}
$OutputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputFullPath) {
    throw "Refusing to overwrite existing archive: $OutputFullPath"
}

$RequiredArtifacts = @(
    (Join-Path $Run0Root "model\run0.safetensors"),
    (Join-Path $Run0Root "model\model_config.json"),
    (Join-Path $Run0Root "model\backbone")
)
if (-not $AllowIncompleteModel) {
    $Missing = @($RequiredArtifacts | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($Missing.Count -gt 0) {
        throw "Model artifact contract is incomplete: $($Missing -join ', ')"
    }
}

$TempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$Stage = Join-Path $TempBase ("deepvoice-run0-" + [guid]::NewGuid().ToString("N"))
$StageFull = [System.IO.Path]::GetFullPath($Stage)
if (-not $StageFull.StartsWith($TempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $StageFull"
}

try {
    New-Item -ItemType Directory -Path $StageFull | Out-Null
    foreach ($Name in @("script.py", "config.toml", "requirements.txt")) {
        Copy-Item -LiteralPath (Join-Path $Run0Root $Name) -Destination $StageFull
    }
    Copy-Item -LiteralPath (Join-Path $Run0Root "deepvoice_run0") -Destination $StageFull -Recurse
    Copy-Item -LiteralPath (Join-Path $Run0Root "model") -Destination $StageFull -Recurse
    Get-ChildItem -LiteralPath $StageFull -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    New-Item -ItemType Directory -Path (Split-Path $OutputFullPath -Parent) -Force | Out-Null
    Compress-Archive -Path (Join-Path $StageFull "*") -DestinationPath $OutputFullPath -CompressionLevel Optimal
    $Archive = Get-Item -LiteralPath $OutputFullPath
    if ($Archive.Length -ge 9GB) {
        throw "Archive exceeds the internal 9 GiB gate: $($Archive.Length) bytes"
    }
    Write-Output "Created $OutputFullPath ($($Archive.Length) bytes)"
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
