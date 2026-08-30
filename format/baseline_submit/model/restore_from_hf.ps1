[CmdletBinding()]
param(
    [string]$RepoId = "datahj/deepvoice2608",
    [string]$Revision = "8671000ad992b8e9e58558c6869126fb7b635a88",
    [string]$DestinationRoot = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

$entries = @(
    [pscustomobject]@{
        HubPath = "storage/df_arena_1b/pytorch_model.bin"
        LocalPath = "df_arena_1b/pytorch_model.bin"
        Bytes = [int64]4591794734
        Sha256 = "780bc14fd4c15e65d58efdef728427cf03cd29cd60be528e97badf8c89087988"
    },
    [pscustomobject]@{
        HubPath = "storage/htdemucs/955717e8-8726e21a.th"
        LocalPath = "htdemucs/955717e8-8726e21a.th"
        Bytes = [int64]84141911
        Sha256 = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
    },
    [pscustomobject]@{
        HubPath = "storage/panns/Cnn14_mAP=0.431.pth"
        LocalPath = "panns/Cnn14_mAP=0.431.pth"
        Bytes = [int64]327428481
        Sha256 = "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
    }
)

$destinationFull = [System.IO.Path]::GetFullPath($DestinationRoot)
New-Item -ItemType Directory -Path $destinationFull -Force | Out-Null

$tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$staging = Join-Path $tempRoot ("deepvoice2608-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $staging | Out-Null

try {
    & hf download $RepoId --revision $Revision --include "storage/**" --local-dir $staging --format quiet
    if ($LASTEXITCODE -ne 0) {
        throw "hf download failed with exit code $LASTEXITCODE"
    }

    foreach ($entry in $entries) {
        $downloaded = Join-Path $staging $entry.HubPath
        if (-not (Test-Path -LiteralPath $downloaded -PathType Leaf)) {
            throw "Missing downloaded file: $($entry.HubPath)"
        }

        $downloadedItem = Get-Item -LiteralPath $downloaded
        if ($downloadedItem.Length -ne $entry.Bytes) {
            throw "Size mismatch for $($entry.HubPath): $($downloadedItem.Length)"
        }

        $downloadedHash = (Get-FileHash -LiteralPath $downloaded -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($downloadedHash -ne $entry.Sha256) {
            throw "SHA-256 mismatch for $($entry.HubPath): $downloadedHash"
        }

        $target = [System.IO.Path]::GetFullPath((Join-Path $destinationFull $entry.LocalPath))
        if (-not $target.StartsWith($destinationFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Target escaped destination root: $target"
        }

        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $existingItem = Get-Item -LiteralPath $target
            $existingHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($existingItem.Length -ne $entry.Bytes -or $existingHash -ne $entry.Sha256) {
                throw "Refusing to overwrite a different local file: $target"
            }
            Write-Host "Already verified: $target"
            continue
        }

        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $downloaded -Destination $target
        Write-Host "Restored: $target"
    }
}
finally {
    $stagingFull = [System.IO.Path]::GetFullPath($staging)
    $stagingLeaf = Split-Path -Leaf $stagingFull
    if ($stagingFull.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        $stagingLeaf.StartsWith("deepvoice2608-", [System.StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $stagingFull -Recurse -Force -ErrorAction SilentlyContinue
    }
}
