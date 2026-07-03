# DiamondEdge — start both servers
# Usage: .\start.ps1

Write-Host "Starting DiamondEdge..." -ForegroundColor Cyan

# Kill anything on 8000 / 5173
$ports = @(8000, 5173)
foreach ($port in $ports) {
    $processId = (netstat -ano | Select-String ":$port " | Select-String "LISTENING" | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1)
    if ($processId) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Write-Host "  Cleared port $port (pid $processId)"
    }
}

Start-Sleep 1

# Start FastAPI in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python -m uvicorn backend.main:app --reload --port 8000" -WindowStyle Normal

Start-Sleep 2

# Start Vite in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; npm run dev" -WindowStyle Normal

Write-Host ""
Write-Host "Both servers starting." -ForegroundColor Green
Write-Host "Open http://localhost:5173 once you see 'VITE ready' in the Vite window." -ForegroundColor Yellow
