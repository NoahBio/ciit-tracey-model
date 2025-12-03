@echo off
REM Smoke test for RL training pipeline
REM Runs minimal training to verify setup works end-to-end
REM
REM Usage:
REM   scripts\smoke_test.bat

echo =====================================
echo Running Training Pipeline Smoke Test
echo =====================================

REM Run integration tests
echo.
echo Running integration tests...
pytest tests\test_training_integration.py -v

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Integration tests failed
    exit /b 1
)

REM Create temp directory
set TEMP_DIR=%TEMP%\ciit_smoke_test_%RANDOM%
mkdir %TEMP_DIR%
mkdir %TEMP_DIR%\models
mkdir %TEMP_DIR%\logs

echo.
echo Test directory: %TEMP_DIR%
echo.

REM Create minimal test config
(
echo experiment_name: smoke_test
echo patterns:
echo   - cold_stuck
echo mechanism: bond_only
echo threshold: 0.9
echo max_sessions: 20
echo total_timesteps: 1000
echo n_envs: 2
echo batch_size: 32
echo n_epochs: 2
echo hidden_size: 64
echo log_dir: %TEMP_DIR%\logs
echo save_freq: 500
echo eval_freq: 500
echo eval_episodes: 5
echo seed: 42
) > %TEMP_DIR%\test_config.yaml

echo Running minimal training...
python run_RL_experiment.py ^
    --config %TEMP_DIR%\test_config.yaml ^
    --output-dir %TEMP_DIR%\models ^
    --no-eval

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed
    rmdir /s /q %TEMP_DIR%
    exit /b 1
)

REM Cleanup
rmdir /s /q %TEMP_DIR%

echo.
echo =====================================
echo ✓ Smoke test PASSED
echo =====================================
echo.
echo Your training pipeline is working correctly!
echo You can now run full experiments with confidence.
