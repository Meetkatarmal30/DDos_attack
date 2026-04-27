@echo off

echo Installing required Python packages...
pip install -r requirements.txt

echo.
echo ==============================================
echo Generating Mock CICIDS2017 Format Dataset
echo ==============================================
python generate_data.py

echo.
echo ==============================================
echo Training Hybrid Machine Learning Model
echo (Wait while SMOTE and RF optimizations run)
echo ==============================================
python train_model.py

echo.
echo ==============================================
echo Firing up Flask Dashboard in new window...
echo ==============================================
start cmd /k "python app.py"

echo.
echo Waiting 3 seconds for server to initialize...
timeout /t 3 /nobreak >nul

echo.
echo ==============================================
echo Initializing Real Time Simulation!
echo Check your Flask Dashboard at http://127.0.0.1:5000
echo ==============================================
python realtime.py

pause
