param(
    [ValidateSet(48, 96, 192, 384)]
    [int[]]$Sizes = @(48, 96, 192, 384),

    [ValidateRange(1, 100000)]
    [int]$Iterations = 5,

    [ValidateSet(3, 4)]
    [int]$MGLevels = 3,

    [ValidateRange(10, 1000)]
    [int]$SampleIntervalMs = 20,

    [ValidateRange(2, 30)]
    [int]$BaselineSeconds = 5,

    [ValidateRange(0, 15)]
    [int]$GpuIndex = 0,

    [string]$MatlabExe = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe',

    [ValidateSet('Both', 'Host', 'GPU')]
    [string]$Metric = 'Both',

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
if ($Metric -eq 'Both') {
    $forward = @{}
    foreach ($key in $PSBoundParameters.Keys) { $forward[$key] = $PSBoundParameters[$key] }
    $forward.Metric = 'Host'
    & $PSCommandPath @forward
    $forward.Metric = 'GPU'
    & $PSCommandPath @forward
    return
}
$resultDir = Join-Path $workspace ('out\memory\memory_' + $Metric.ToLower() +
    '_n' + $Iterations + '_L' + $MGLevels)
$baseResultDir = $resultDir
$repeatIndex = 1
while (Test-Path -LiteralPath $resultDir) {
    $resultDir = $baseResultDir + '_t' + $repeatIndex
    $repeatIndex++
}

if (-not (Test-Path -LiteralPath $MatlabExe -PathType Leaf)) {
    throw "MATLAB executable not found: $MatlabExe"
}
$nvidiaSmi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($nvidiaSmi)) {
    throw 'nvidia-smi was not found on PATH.'
}

function Convert-ToMatlabPath {
    param([Parameter(Mandatory)][string]$Path)
    return $Path.Replace('\', '/').Replace("'", "''")
}

function Get-Median {
    param([Parameter(Mandatory)][double[]]$Values)
    if ($Values.Count -eq 0) { throw 'Cannot compute a median from zero samples.' }
    $sorted = @($Values | Sort-Object)
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($Values.Count % 2) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2
}

function Get-GpuRecords {
    param([Parameter(Mandatory)][string]$Path)
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $records = [System.Collections.Generic.List[object]]::new()
    foreach ($line in Get-Content -LiteralPath $Path) {
        $parts = $line -split ',', 2
        if ($parts.Count -ne 2) { continue }
        $sampleTime = [datetime]::MinValue
        $usedMiB = 0.0
        $validTime = [datetime]::TryParseExact(
            $parts[0].Trim(), 'yyyy/MM/dd HH:mm:ss.fff', $culture,
            [Globalization.DateTimeStyles]::None, [ref]$sampleTime)
        $validMemory = [double]::TryParse(
            $parts[1].Trim(), [Globalization.NumberStyles]::Float,
            $culture, [ref]$usedMiB)
        if ($validTime -and $validMemory) {
            $records.Add([pscustomobject]@{
                SampleTime = $sampleTime
                Timestamp  = $sampleTime.ToString('yyyy/MM/dd HH:mm:ss.fff')
                UsedMiB    = $usedMiB
            })
        }
    }
    return $records
}

$matlabWorkspace = Convert-ToMatlabPath $workspace
$summaryRows = [System.Collections.Generic.List[object]]::new()
$summaryFile = Join-Path $resultDir 'summary.csv'

Write-Host 'End-to-end GPU topology optimization memory benchmark'
Write-Host "Selected metric    : $Metric (independent MATLAB run)"
Write-Host 'Mesh X sizes       :' ($Sizes -join ', ')
Write-Host "MG levels          : $MGLevels"
Write-Host 'ft / V0 / tol      : 2 / 0.12 / 1e-3'
Write-Host "Iterations         : $Iterations"
Write-Host "Sampling interval  : $SampleIntervalMs ms"
Write-Host "Result directory   : $resultDir"

foreach ($xSize in $Sizes) {
    $ySize = [int]($xSize / 2)
    $zSize = $ySize
    $caseName = "${xSize}x${ySize}x${zSize}"
    $caseDir = Join-Path $resultDir $caseName
    $readyMarker = Join-Path $caseDir 'matlab_ready.flag'
    $startMarker = Join-Path $caseDir 'optimization_start.flag'
    $endMarker = Join-Path $caseDir 'optimization_end.flag'
    $gpuRawFile = Join-Path $caseDir 'gpu_memory_raw.csv'
    $hostRawFile = Join-Path $caseDir 'host_working_set_raw.csv'
    $matlabStdout = Join-Path $caseDir 'matlab.stdout.log'
    $matlabStderr = Join-Path $caseDir 'matlab.stderr.log'
    $nvidiaStderr = Join-Path $caseDir 'nvidia_smi.stderr.log'

    $readyPath = Convert-ToMatlabPath $readyMarker
    $startPath = Convert-ToMatlabPath $startMarker
    $endPath = Convert-ToMatlabPath $endMarker
    $batchCommand = "cd('$matlabWorkspace'); " +
        "fid=fopen('$readyPath','w'); assert(fid>=0); fclose(fid); " +
        "pause($BaselineSeconds); " +
        "fid=fopen('$startPath','w'); assert(fid>=0); fclose(fid); " +
        "TOP_GPU(true($ySize,$xSize,$zSize),'optCase',1,'V0',0.12," +
        "'rMin',sqrt(3),'nLoop',$Iterations,'ft',2,'filter_method','pde'," +
        "'mixed_Precision',0,'super_element',0,'tol',1e-3,'numLevels',$MGLevels); " +
        "fid=fopen('$endPath','w'); assert(fid>=0); fclose(fid);"

    Write-Host "`n[$caseName]"
    if ($DryRun) {
        Write-Host "MATLAB COMMAND: $batchCommand"
        continue
    }

    New-Item -ItemType Directory -Path $caseDir -Force | Out-Null
    $existingMatlabIds = @(Get-Process -Name MATLAB -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Id)
    $hostRecords = [System.Collections.Generic.List[object]]::new()
    $nvidiaProcess = $null
    $matlabProcess = $null
    $matlabEndTime = $null

    try {
        $nvidiaProcess = Start-Process -FilePath $nvidiaSmi -ArgumentList @(
            '--query-gpu=timestamp,memory.used',
            '--format=csv,noheader,nounits',
            '-i', $GpuIndex.ToString(),
            '-lms', $SampleIntervalMs.ToString()
        ) -PassThru -WindowStyle Hidden -RedirectStandardOutput $gpuRawFile `
            -RedirectStandardError $nvidiaStderr

        $matlabProcess = Start-Process -FilePath $MatlabExe `
            -ArgumentList @('-batch', "`"$batchCommand`"") `
            -PassThru -WindowStyle Hidden -RedirectStandardOutput $matlabStdout `
            -RedirectStandardError $matlabStderr

        Write-Host "MATLAB PID         : $($matlabProcess.Id)"
        while ($true) {
            $matlabProcess.Refresh()
            if ($matlabProcess.HasExited) { break }
            $newMatlabProcesses = @(Get-Process -Name MATLAB -ErrorAction SilentlyContinue |
                Where-Object { $existingMatlabIds -notcontains $_.Id })
            if ($newMatlabProcesses.Count -gt 0) {
                $worker = $newMatlabProcesses |
                    Sort-Object WorkingSet64 -Descending | Select-Object -First 1
                $hostRecords.Add([pscustomobject]@{
                    SampleTime     = Get-Date
                    Timestamp      = Get-Date -Format 'yyyy/MM/dd HH:mm:ss.fff'
                    ProcessId      = $worker.Id
                    WorkingSetMiB  = [double]$worker.WorkingSet64 / 1MB
                })
            }
            Start-Sleep -Milliseconds $SampleIntervalMs
        }
        $matlabEndTime = Get-Date
        $matlabProcess.WaitForExit()
    }
    finally {
        if ($null -ne $nvidiaProcess) {
            $nvidiaProcess.Refresh()
            if (-not $nvidiaProcess.HasExited) {
                Stop-Process -Id $nvidiaProcess.Id -Force
                $nvidiaProcess.WaitForExit()
            }
        }
    }

    Start-Sleep -Milliseconds 200
    if (-not (Test-Path -LiteralPath $readyMarker) -or
        -not (Test-Path -LiteralPath $startMarker)) {
        throw "MATLAB did not reach the optimization start marker for $caseName. Check $matlabStdout and $matlabStderr."
    }
    $readyTime = (Get-Item -LiteralPath $readyMarker).LastWriteTime
    $startTime = (Get-Item -LiteralPath $startMarker).LastWriteTime
    $endTime = if (Test-Path -LiteralPath $endMarker) {
        (Get-Item -LiteralPath $endMarker).LastWriteTime
    } else { $matlabEndTime }

    $gpuRecords = @(Get-GpuRecords $gpuRawFile)
    $hostRecords | Select-Object Timestamp,ProcessId,WorkingSetMiB |
        Export-Csv -LiteralPath $hostRawFile -NoTypeInformation -Encoding UTF8
    $hostBaseline = @($hostRecords | Where-Object {
        $_.SampleTime -ge $readyTime -and $_.SampleTime -lt $startTime })
    $hostDuring = @($hostRecords | Where-Object {
        $_.SampleTime -ge $startTime -and $_.SampleTime -le $endTime })
    $gpuBaseline = @($gpuRecords | Where-Object {
        $_.SampleTime -ge $readyTime -and $_.SampleTime -lt $startTime })
    $gpuDuring = @($gpuRecords | Where-Object {
        $_.SampleTime -ge $startTime -and $_.SampleTime -le $endTime })
    if ($hostBaseline.Count -eq 0 -or $hostDuring.Count -eq 0 -or
        $gpuBaseline.Count -eq 0 -or $gpuDuring.Count -eq 0) {
        throw "Missing baseline or run samples for $caseName. Check raw CSV files in $caseDir."
    }

    $hostBaselineMiB = Get-Median ([double[]]@($hostBaseline.WorkingSetMiB))
    $gpuBaselineMiB = Get-Median ([double[]]@($gpuBaseline.UsedMiB))
    $hostPeakMiB = [double](($hostDuring | Measure-Object WorkingSetMiB -Maximum).Maximum)
    $gpuPeakMiB = [double](($gpuDuring | Measure-Object UsedMiB -Maximum).Maximum)
    $row = [pscustomobject]@{
        MeshResolution          = $caseName
        Elements                = [int64]$xSize * $ySize * $zSize
        MGLevels                = $MGLevels
        Iterations              = $Iterations
        SelectedMetric          = $Metric
        HostWorkingSetGiB       = if ($Metric -eq 'Host') { ($hostPeakMiB - $hostBaselineMiB) / 1024 } else { [double]::NaN }
        GPUMemoryGiB            = if ($Metric -eq 'GPU') { ($gpuPeakMiB - $gpuBaselineMiB) / 1024 } else { [double]::NaN }
        HostBaselineGiB         = $hostBaselineMiB / 1024
        HostAbsolutePeakGiB     = $hostPeakMiB / 1024
        GPUBaselineGiB          = $gpuBaselineMiB / 1024
        GPUAbsolutePeakGiB      = $gpuPeakMiB / 1024
        HostBaselineSamples     = $hostBaseline.Count
        HostRunSamples          = $hostDuring.Count
        GPUBaselineSamples      = $gpuBaseline.Count
        GPURunSamples           = $gpuDuring.Count
        MatlabExitCode          = $matlabProcess.ExitCode
    }
    $summaryRows.Add($row)
    $summaryRows | Export-Csv -LiteralPath $summaryFile -NoTypeInformation -Encoding UTF8
    $row | Select-Object MeshResolution,HostWorkingSetGiB,GPUMemoryGiB,MatlabExitCode |
        Format-Table -AutoSize
    if ($matlabProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $endMarker)) {
        throw "MATLAB failed for $caseName. Check $matlabStdout and $matlabStderr."
    }
}

if (-not $DryRun) {
    Write-Host "Summary: $summaryFile"
}
