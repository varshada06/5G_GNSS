@echo off
echo ============================================================
echo GNSS-5G Hybrid Positioning with AI-Enhanced MRAKF
echo ============================================================
echo.

echo Step 1: Generating dataset...
python generate_data.py

echo.
echo Step 2: Running main project...
python main_project.py

echo.
echo Step 3: Project complete!
pause