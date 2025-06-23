#!/bin/bash

set -e

# 1. Update and upgrade system
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# 2. Install required system packages
echo "Installing system dependencies..."
sudo apt install -y python3-opencv python3-pip python3-picamera2 python3-libcamera libatlas-base-dev i2c-tools python3-venv

# 3. Enable I2C interface
echo "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# 4. Reboot check
echo ""
echo "If this is your first time enabling I2C, you should reboot your Raspberry Pi after this script."
echo ""

# 5. Create Python virtual environment if not already present
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv --system-site-packages
fi

# 6. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 7. Upgrade pip in venv
echo "Upgrading pip..."
pip install --upgrade pip

# 8. Install Python requirements
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found. Please create it before running this script."
    deactivate
    exit 1
fi

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete! You are ready to run your hand-tracking script."
echo "To activate the virtual environment next time, run: source venv/bin/activate"
echo ""