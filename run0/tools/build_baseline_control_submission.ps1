[CmdletBinding()]
param(
    [string]$SourcePath = "",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$Run0Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkspaceRoot = (Resolve-Path (Join-Path $Run0Root "..")).Path
if (-not $SourcePath) {
    $SourcePath = Join-Path $WorkspaceRoot "format\baseline_submit"
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $Run0Root "build\submit-baseline-control.zip"
}

$SourceFullPath = (Resolve-Path -LiteralPath $SourcePath).Path
$OutputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$ManifestFullPath = [System.IO.Path]::ChangeExtension(
    $OutputFullPath,
    ".manifest.json"
)

if (Test-Path -LiteralPath $OutputFullPath) {
    throw "Refusing to overwrite existing archive: $OutputFullPath"
}
if (Test-Path -LiteralPath $ManifestFullPath) {
    throw "Refusing to overwrite existing manifest: $ManifestFullPath"
}

$RequiredRootItems = @("model", "script.py", "requirements.txt")
foreach ($Name in $RequiredRootItems) {
    $Path = Join-Path $SourceFullPath $Name
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Submission source is missing required root item: $Path"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $SourceFullPath "model") -PathType Container)) {
    throw "Submission source model item is not a directory"
}

$UnexpectedRootItems = @(
    Get-ChildItem -LiteralPath $SourceFullPath -Force |
        Where-Object { $_.Name -notin $RequiredRootItems }
)
if ($UnexpectedRootItems.Count -gt 0) {
    throw "Unexpected root items would violate the submission contract: $($UnexpectedRootItems.Name -join ', ')"
}

# Verify the three large checkpoints before packaging them. SHA256SUMS.txt is
# part of the reviewed organizer baseline and uses paths relative to model/.
$ModelRoot = Join-Path $SourceFullPath "model"
$HashListPath = Join-Path $ModelRoot "SHA256SUMS.txt"
if (-not (Test-Path -LiteralPath $HashListPath -PathType Leaf)) {
    throw "Missing model hash list: $HashListPath"
}

$VerifiedModelHashes = [ordered]@{}
foreach ($Line in Get-Content -LiteralPath $HashListPath) {
    if ($Line -notmatch '^([0-9a-fA-F]{64})\s+(.+)$') {
        if ($Line.Trim()) {
            throw "Invalid SHA256SUMS line: $Line"
        }
        continue
    }

    $ExpectedHash = $Matches[1].ToLowerInvariant()
    $RelativePath = $Matches[2].Trim().Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    $ModelPath = Join-Path $ModelRoot $RelativePath
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
        throw "Hashed model file is missing: $ModelPath"
    }
    $ActualHash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $ExpectedHash) {
        throw "Model hash mismatch for $RelativePath"
    }
    $VerifiedModelHashes[$Matches[2].Trim()] = $ActualHash
}
if ($VerifiedModelHashes.Count -eq 0) {
    throw "No model hashes were verified"
}

$Files = @(
    Get-ChildItem -LiteralPath $SourceFullPath -Recurse -Force -File |
        Sort-Object FullName
)
if ($Files.Count -eq 0) {
    throw "Submission source contains no files"
}

$OutputDirectory = Split-Path $OutputFullPath -Parent
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$PartialPath = $OutputFullPath + ".partial-" + [guid]::NewGuid().ToString("N")

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

try {
    $FileStream = [System.IO.File]::Open(
        $PartialPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    try {
        $Archive = New-Object System.IO.Compression.ZipArchive(
            $FileStream,
            [System.IO.Compression.ZipArchiveMode]::Create,
            $false
        )
        try {
            $ModelDirectoryEntry = $Archive.CreateEntry("model/")
            $ModelDirectoryEntry.LastWriteTime = [System.DateTimeOffset]::Parse(
                "1980-01-01T00:00:00Z"
            )

            foreach ($File in $Files) {
                $RelativePath = $File.FullName.Substring($SourceFullPath.Length)
                $RelativePath = $RelativePath.TrimStart([char[]]@('\', '/'))
                $EntryName = $RelativePath.Replace('\', '/')
                if ($EntryName.StartsWith('/') -or $EntryName.Contains('../')) {
                    throw "Unsafe archive entry name: $EntryName"
                }

                # Checkpoints are already binary-compressed. Store mode makes a
                # deterministic Zip64 archive without spending minutes deflating them.
                $Entry = $Archive.CreateEntry(
                    $EntryName,
                    [System.IO.Compression.CompressionLevel]::NoCompression
                )
                $Entry.LastWriteTime = [System.DateTimeOffset]::Parse(
                    "1980-01-01T00:00:00Z"
                )
                $InputStream = $File.OpenRead()
                try {
                    $EntryStream = $Entry.Open()
                    try {
                        $InputStream.CopyTo($EntryStream, 4MB)
                    }
                    finally {
                        $EntryStream.Dispose()
                    }
                }
                finally {
                    $InputStream.Dispose()
                }
            }
        }
        finally {
            $Archive.Dispose()
        }
    }
    finally {
        $FileStream.Dispose()
    }

    Move-Item -LiteralPath $PartialPath -Destination $OutputFullPath
}
finally {
    if (Test-Path -LiteralPath $PartialPath) {
        Remove-Item -LiteralPath $PartialPath -Force
    }
}

# Re-open the final file and validate the same shape the evaluator sees.
$ReadArchive = [System.IO.Compression.ZipFile]::OpenRead($OutputFullPath)
try {
    $Entries = @($ReadArchive.Entries)
    $EntryNames = @($Entries | ForEach-Object { $_.FullName })
    foreach ($RequiredEntry in @("script.py", "requirements.txt", "model/")) {
        if ($RequiredEntry -notin $EntryNames) {
            throw "Archive is missing required entry: $RequiredEntry"
        }
    }

    $UnsafeEntries = @(
        $EntryNames | Where-Object {
            $_.StartsWith('/') -or $_.Contains('\\') -or $_.Contains('../')
        }
    )
    if ($UnsafeEntries.Count -gt 0) {
        throw "Archive contains unsafe paths: $($UnsafeEntries -join ', ')"
    }

    $UnexpectedEntries = @(
        $EntryNames | Where-Object {
            $_ -ne "script.py" -and
            $_ -ne "requirements.txt" -and
            -not $_.StartsWith("model/")
        }
    )
    if ($UnexpectedEntries.Count -gt 0) {
        throw "Archive contains unexpected top-level content: $($UnexpectedEntries -join ', ')"
    }

    $ArchiveBytes = (Get-Item -LiteralPath $OutputFullPath).Length
    $UncompressedBytes = [long](
        $Entries | Measure-Object -Property Length -Sum
    ).Sum
    if ($ArchiveBytes -ge 10GB) {
        throw "Archive exceeds the official 10 GB limit: $ArchiveBytes bytes"
    }
    if ($UncompressedBytes -ge 32GB) {
        throw "Archive exceeds the official 32 GB expanded limit: $UncompressedBytes bytes"
    }
}
finally {
    $ReadArchive.Dispose()
}

$ArchiveHash = (Get-FileHash -LiteralPath $OutputFullPath -Algorithm SHA256).Hash.ToLowerInvariant()
$GitCommit = (& git -C $WorkspaceRoot rev-parse HEAD).Trim()
$Manifest = [ordered]@{
    artifact = "dacon-deepvoice-baseline-control"
    purpose = "leaderboard baseline control; not a trained Run0 detector"
    archive = [System.IO.Path]::GetFileName($OutputFullPath)
    archive_sha256 = $ArchiveHash
    archive_bytes = $ArchiveBytes
    uncompressed_bytes = $UncompressedBytes
    entry_count = $Entries.Count
    source = "format/baseline_submit"
    source_git_commit = $GitCommit
    model_hashes = $VerifiedModelHashes
    run0_model_ready = $false
    validations = [ordered]@{
        model_sha256 = $true
        required_root_entries = $true
        no_extra_top_level_directory = $true
        offline_models_included = $true
        archive_under_10gb = $true
        expanded_under_32gb = $true
    }
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestFullPath -Encoding UTF8

Write-Output "Created baseline control submission: $OutputFullPath"
Write-Output "SHA-256: $ArchiveHash"
Write-Output "Archive bytes: $ArchiveBytes"
Write-Output "Expanded bytes: $UncompressedBytes"
Write-Output "Manifest: $ManifestFullPath"
