@echo off
echo Creating Python 3.11 virtual environment...
python -m venv venv_cuda

echo Activating virtual environment...
call venv_cuda\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo Installing requirements...
pip install -r requirements.txt

echo Creating necessary directories...
mkdir album 2>nul
mkdir clustered_faces 2>nul
mkdir face_search_results 2>nul
mkdir uploads 2>nul

echo Setup complete!
echo To activate the environment, run: venv_cuda\Scripts\activate