#!/bin/bash

# Script to install visualization dependencies for SNNFW
# This installs the X11 development libraries required by GLFW

echo "Installing visualization dependencies..."

# Check if running on Ubuntu/Debian
if [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu system"
    echo "Installing X11 development libraries..."
    
    sudo apt-get update
    sudo apt-get install -y \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libx11-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev
    
    echo "Dependencies installed successfully!"
else
    echo "This script is designed for Ubuntu/Debian systems."
    echo "For other systems, please install the following manually:"
    echo "  - X11 development headers"
    echo "  - XRandR development headers"
    echo "  - Xinerama development headers"
    echo "  - XCursor development headers"
    echo "  - Xi development headers"
    echo "  - OpenGL development headers"
fi

