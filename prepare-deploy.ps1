# PowerShell script to prepare for deployment
Write-Host "=== Preparing LoL Analytics for deployment ===" -ForegroundColor Cyan

# Copy models to api folder
Write-Host ""
Write-Host "[1/3] Copying ML models to api folder..." -ForegroundColor Yellow

if (Test-Path "ml\models") {
    if (Test-Path "api\models") { Remove-Item -Recurse -Force "api\models" }
    Copy-Item -Recurse "ml\models" "api\models"
    Write-Host "[OK] Copied ml\models -> api\models" -ForegroundColor Green
} else {
    Write-Host "[ERROR] ml\models not found! Run training first." -ForegroundColor Red
}

# Copy data to api folder
Write-Host "[2/3] Copying stats data to api folder..." -ForegroundColor Yellow

if (Test-Path "ml\data") {
    if (Test-Path "api\data") { Remove-Item -Recurse -Force "api\data" }
    Copy-Item -Recurse "ml\data" "api\data"
    Write-Host "[OK] Copied ml\data -> api\data" -ForegroundColor Green
} else {
    Write-Host "[ERROR] ml\data not found! Run training first." -ForegroundColor Red
}

# Copy raw data to api folder
Write-Host "[3/3] Copying raw data to api folder..." -ForegroundColor Yellow

$rawDataFile = "2025_LoL_esports_match_data_from_OraclesElixir.csv"
if (Test-Path $rawDataFile) {
    Copy-Item -Force $rawDataFile "api\$rawDataFile"
    $sizeMB = [math]::Round((Get-Item "api\$rawDataFile").Length / 1MB, 1)
    Write-Host "[OK] Copied $rawDataFile ($sizeMB MB)" -ForegroundColor Green
} else {
    Write-Host "[WARN] Raw data not found - some features will be limited" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Deployment checklist ===" -ForegroundColor Cyan
Write-Host "1. Push code to GitHub" -ForegroundColor White
Write-Host "2. Deploy API to Railway (railway.app)" -ForegroundColor White
Write-Host "3. Deploy Web to Vercel (vercel.com)" -ForegroundColor White
Write-Host "4. Set NEXT_PUBLIC_API_URL in Vercel" -ForegroundColor White

Write-Host ""
Write-Host "See DEPLOY.md for detailed instructions" -ForegroundColor Yellow
Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
