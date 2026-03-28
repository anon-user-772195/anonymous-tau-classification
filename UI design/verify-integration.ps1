# NeuroFoldNet Integration Verification Script
# Run this in PowerShell from your project root

Write-Host "===============================" -ForegroundColor Cyan
Write-Host "NeuroFoldNet Integration Check" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check 1: lib/api.ts exists
Write-Host "1. Checking lib/api.ts..." -NoNewline
if (Test-Path "lib/api.ts") {
    $size = (Get-Item "lib/api.ts").Length
    Write-Host " OK - EXISTS (Size: $size bytes)" -ForegroundColor Green
    
    # Check if it has the debug logs
    $content = Get-Content "lib/api.ts" -Raw
    if ($content -match "console\.log") {
        Write-Host "   OK - Contains debug logs" -ForegroundColor Green
    } else {
        Write-Host "   ERROR - Missing debug logs - file not updated!" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host " ERROR - MISSING!" -ForegroundColor Red
    Write-Host "   You need to create lib/api.ts" -ForegroundColor Yellow
    $allGood = $false
}

# Check 2: .env.local exists
Write-Host "2. Checking .env.local..." -NoNewline
if (Test-Path ".env.local") {
    Write-Host " OK - EXISTS" -ForegroundColor Green
    $envContent = Get-Content ".env.local" -Raw
    if ($envContent -match "localhost:5000") {
        Write-Host "   OK - Correct API URL" -ForegroundColor Green
    } else {
        Write-Host "   ERROR - Wrong API URL!" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host " ERROR - MISSING!" -ForegroundColor Red
    Write-Host "   You need to create .env.local" -ForegroundColor Yellow
    $allGood = $false
}

# Check 3: InferencePanel.tsx updated
Write-Host "3. Checking InferencePanel.tsx..." -NoNewline
if (Test-Path "components/InferencePanel.tsx") {
    $size = (Get-Item "components/InferencePanel.tsx").Length
    Write-Host " OK - EXISTS (Size: $size bytes)" -ForegroundColor Green
    
    $content = Get-Content "components/InferencePanel.tsx" -Raw
    if ($content -match 'apiClient') {
        Write-Host "   OK - Imports apiClient" -ForegroundColor Green
    } else {
        Write-Host "   ERROR - Still using old version!" -ForegroundColor Red
        Write-Host "   You need to replace this file with the updated version" -ForegroundColor Yellow
        $allGood = $false
    }
    
    if ($content -match 'backendHealthy') {
        Write-Host "   OK - Has backend health check" -ForegroundColor Green
    } else {
        Write-Host "   ERROR - Missing backend integration!" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Host " ERROR - MISSING!" -ForegroundColor Red
    $allGood = $false
}

# Check 4: BatchPanel.tsx updated
Write-Host "4. Checking BatchPanel.tsx..." -NoNewline
if (Test-Path "components/BatchPanel.tsx") {
    $size = (Get-Item "components/BatchPanel.tsx").Length
    Write-Host " OK - EXISTS (Size: $size bytes)" -ForegroundColor Green
    
    $content = Get-Content "components/BatchPanel.tsx" -Raw
    if ($content -match 'apiClient') {
        Write-Host "   OK - Imports apiClient" -ForegroundColor Green
    } else {
        Write-Host "   ERROR - Still using old version!" -ForegroundColor Red
        Write-Host "   You need to replace this file" -ForegroundColor Yellow
        $allGood = $false
    }
} else {
    Write-Host " ERROR - MISSING!" -ForegroundColor Red
    $allGood = $false
}

# Check 5: AboutPanel.tsx updated
Write-Host "5. Checking AboutPanel.tsx..." -NoNewline
if (Test-Path "components/AboutPanel.tsx") {
    $size = (Get-Item "components/AboutPanel.tsx").Length
    Write-Host " OK - EXISTS (Size: $size bytes)" -ForegroundColor Green
    
    $content = Get-Content "components/AboutPanel.tsx" -Raw
    if ($content -match 'apiClient') {
        Write-Host "   OK - Imports apiClient" -ForegroundColor Green
    } else {
        Write-Host "   ERROR - Still using old version!" -ForegroundColor Red
        Write-Host "   You need to replace this file" -ForegroundColor Yellow
        $allGood = $false
    }
} else {
    Write-Host " ERROR - MISSING!" -ForegroundColor Red
    $allGood = $false
}

# Check 6: Flask model file
Write-Host "6. Checking Flask model..." -NoNewline
if (Test-Path "flask_backend/neurofoldnet_model.pkl") {
    $size = [math]::Round((Get-Item "flask_backend/neurofoldnet_model.pkl").Length / 1MB, 2)
    Write-Host " OK - EXISTS (Size: $size MB)" -ForegroundColor Green
} else {
    Write-Host " ERROR - MISSING!" -ForegroundColor Red
    Write-Host "   Copy your trained model to flask_backend/" -ForegroundColor Yellow
    $allGood = $false
}

Write-Host ""
Write-Host "===============================" -ForegroundColor Cyan

if ($allGood) {
    Write-Host "SUCCESS - ALL CHECKS PASSED!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Stop Next.js (Ctrl+C)" -ForegroundColor White
    Write-Host "2. Delete .next folder: Remove-Item -Recurse -Force .next" -ForegroundColor White
    Write-Host "3. Restart Next.js: npm run dev" -ForegroundColor White
    Write-Host "4. Hard refresh browser: Ctrl+Shift+R" -ForegroundColor White
} else {
    Write-Host "FAILED - SOME CHECKS FAILED!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please fix the issues above before proceeding." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "If you see 'ERROR - Still using old version':" -ForegroundColor Yellow
    Write-Host "  You need to REPLACE the component files with the updated versions" -ForegroundColor White
    Write-Host "  from the outputs I provided." -ForegroundColor White
}

Write-Host "===============================" -ForegroundColor Cyan