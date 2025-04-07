#!/bin/bash
# Face Recognition API CUDA Setup Script for Ubuntu
# This script sets up just the Python virtual environment with CUDA support

set -e  # Exit on any error

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Python virtual environment with CUDA support${NC}"

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}No NVIDIA GPU detected. CUDA setup may not work correctly.${NC}"
    echo "The script will proceed, but you may need to install NVIDIA drivers first."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Install Python and pip if not already installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Installing Python 3...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# Create project directory if it doesn't exist
PROJECT_DIR="${PWD}/face_recognition_api"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create Python virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv_cuda

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv_cuda/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install main requirements
echo -e "${YELLOW}Installing main dependencies...${NC}"
pip install fastapi>=0.104.0 \
            uvicorn>=0.24.0 \
            pydantic>=2.4.2 \
            pydantic-settings>=2.0.3 \
            pydantic[email]>=2.4.2 \
            pyodbc>=4.0.39 \
            python-multipart>=0.0.6 \
            python-dotenv>=1.0.0 \
            email-validator>=2.0.0 \
            scikit-learn>=1.3.0 \
            passlib>=1.7.4 \
            python-jose>=3.3.0 \
            bcrypt>=4.0.1 \
            opencv-python>=4.8.0 \
            numpy>=1.24.0 \
            insightface==0.7.3 \
            redis>=5.0.0

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p storage
mkdir -p temp
mkdir -p album
mkdir -p clustered_faces
mkdir -p face_search_results
mkdir -p uploads

# Create .env example file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOL
# API Settings
BASE_URL=http://localhost:8000
API_KEY=your_api_key_here

# Storage Settings
STORAGE_DIR=storage
TEMP_DIR=temp

# Database Settings
DB_SERVER=your_db_server
DB_NAME=FaceRecognitionDB
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_DRIVER={ODBC Driver 17 for SQL Server}

# JWT Settings
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS512
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15

# Redis Settings (if used)
REDIS_URL=redis://:password@localhost:6379/0
EOL
fi

echo -e "${GREEN}Virtual environment with CUDA support setup complete!${NC}"
echo -e "${YELLOW}To activate the environment, run:${NC}"
echo "source ${PROJECT_DIR}/venv_cuda/bin/activate"
echo
echo -e "${YELLOW}Note: Make sure to install the ODBC Driver for SQL Server if you need database connectivity:${NC}"
echo "sudo apt-get install -y msodbcsql17 unixodbc-dev"