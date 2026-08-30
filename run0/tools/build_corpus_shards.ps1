[CmdletBinding()]
param(
    [string]$CorpusPath = "",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$Run0Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $CorpusPath) {
    $CorpusPath = Join-Path $Run0Root "data\corpus-v1"
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $Run0Root "build\corpus-shards-v1"
}
$CorpusFull = (Resolve-Path -LiteralPath $CorpusPath).Path
$OutputFull = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputFull) {
    throw "Refusing to merge with or overwrite shard output: $OutputFull"
}
$Tar = (Get-Command tar.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $OutputFull | Out-Null
$Records = @()
foreach ($Index in 0..7) {
    $Code = "C{0:D2}" -f $Index
    $SourceDirectory = Join-Path $CorpusFull "audio\$Code"
    $Files = @(Get-ChildItem -LiteralPath $SourceDirectory -File -Filter "*.wav")
    if ($Files.Count -ne 3000) {
        throw "Expected 3000 WAV files in $SourceDirectory; found $($Files.Count)"
    }
    $ArchiveName = "audio-$Code.tar"
    $ArchivePath = Join-Path $OutputFull $ArchiveName
    & $Tar -cf $ArchivePath -C $CorpusFull "audio/$Code"
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed for $Code with exit code $LASTEXITCODE"
    }
    $Item = Get-Item -LiteralPath $ArchivePath
    $Records += [ordered]@{
        path = "shards/$ArchiveName"
        sha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        bytes = $Item.Length
        files = $Files.Count
    }
}
$ManifestHash = (Get-FileHash -LiteralPath (Join-Path $CorpusFull "manifest.jsonl") -Algorithm SHA256).Hash.ToLowerInvariant()
$Payload = [ordered]@{
    format_version = 1
    corpus_manifest_sha256 = $ManifestHash
    shards = $Records
}
$Json = $Payload | ConvertTo-Json -Depth 5
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $OutputFull "SHARDS.json"), $Json + "`n", $Utf8NoBom)
Write-Output "Created 8 reviewed corpus shards at $OutputFull"
