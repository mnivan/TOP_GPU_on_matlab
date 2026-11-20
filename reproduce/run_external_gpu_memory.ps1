param(
    [ValidateSet(48, 96, 192, 384)][int[]]$Sizes = @(48, 96, 192, 384),
    [ValidateRange(1, 100000)][int]$Iterations = 5,
    [ValidateSet(3, 4)][int]$MGLevels = 3,
    [ValidateRange(10, 1000)][int]$SampleIntervalMs = 20,
    [ValidateRange(2, 30)][int]$BaselineSeconds = 5,
    [ValidateRange(0, 15)][int]$GpuIndex = 0,
    [string]$MatlabExe = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe',
    [switch]$DryRun
)

# Keep the original entry point; GPU peaks use their own MATLAB runs.
& (Join-Path $PSScriptRoot 'run_external_memory.ps1') -Metric GPU @PSBoundParameters
