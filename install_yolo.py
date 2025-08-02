"""
YOLO Installation Script
Installs YOLOv8 for the person detection system
"""

import subprocess
import sys

def install_yolo():
    """Install YOLOv8 and test the installation."""
    print("Installing YOLOv8 for person detection...")
    
    try:
        # Install ultralytics package
        print("1. Installing ultralytics package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✓ ultralytics installed successfully")
        
        # Test YOLO import and download model
        print("2. Testing YOLO import and downloading model...")
        from ultralytics import YOLO
        
        # This will download the YOLOv8n model (~6MB) on first use
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 nano model downloaded successfully")
        
        # Test detection on a dummy image
        import numpy as np
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print("✓ YOLO model tested successfully")
        
        print("\n🎉 YOLO installation completed successfully!")
        print("You can now run: python run_phase1_step2.py")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install ultralytics: {e}")
        return False
    except ImportError as e:
        print(f"❌ Failed to import YOLO: {e}")
        return False
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

if __name__ == "__main__":
    success = install_yolo()
    if not success:
        print("\n🔧 Manual installation:")
        print("pip install ultralytics")
        sys.exit(1)