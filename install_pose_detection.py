"""
Pose Detection Installation Script
Installs YOLOv8 pose model for Phase 2 Step 2 orientation detection
"""

import subprocess
import sys

def install_pose_dependencies():
    """Install pose detection dependencies for orientation estimation."""
    print("Installing pose detection dependencies for Phase 2 Step 2...")
    
    try:
        # Ensure ultralytics is up to date with pose support
        print("1. Updating ultralytics for pose detection...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
        print("✓ ultralytics updated successfully")
        
        # Test pose model download and loading
        print("2. Testing pose model download...")
        from ultralytics import YOLO
        
        # Download YOLOv8 pose model (larger than object detection model)
        print("   Downloading YOLOv8n-pose model (~6MB)...")
        pose_model = YOLO('yolov8n-pose.pt')
        print("✓ YOLOv8n-pose model downloaded successfully")
        
        # Test pose detection on dummy image
        import numpy as np
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = pose_model(dummy_image, verbose=False)
        print("✓ Pose model tested successfully")
        
        # Optional: Download larger pose model for better accuracy
        print("3. Optional: Downloading YOLOv8s-pose for better accuracy...")
        try:
            pose_model_s = YOLO('yolov8s-pose.pt')
            print("✓ YOLOv8s-pose model downloaded successfully")
        except Exception as e:
            print(f"⚠️ YOLOv8s-pose download failed (optional): {e}")
        
        print("\n🎉 Pose detection setup completed successfully!")
        print("You can now run: python run_phase2_step2.py")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install/update ultralytics: {e}")
        return False
    except ImportError as e:
        print(f"❌ Failed to import YOLO: {e}")
        return False
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def check_opencv_pose_support():
    """Check if OpenCV has DNN pose support (alternative method)."""
    try:
        import cv2
        
        # Check OpenCV version
        opencv_version = cv2.__version__
        print(f"OpenCV version: {opencv_version}")
        
        # Check if DNN module is available
        if hasattr(cv2, 'dnn'):
            print("✓ OpenCV DNN module available for alternative pose detection")
            return True
        else:
            print("⚠️ OpenCV DNN module not available")
            return False
            
    except ImportError:
        print("❌ OpenCV not found")
        return False

def show_alternative_methods():
    """Show alternative orientation detection methods if pose fails."""
    print("\n📋 Alternative Orientation Methods Available:")
    print("1. Movement-based orientation (tracks movement direction)")
    print("2. Depth gradient analysis (analyzes body asymmetry)")
    print("3. Manual fallback orientations")
    print("\nThe system will automatically use the best available method.")

if __name__ == "__main__":
    print("Phase 2 Step 2: Pose Detection Setup")
    print("=" * 40)
    
    success = install_pose_dependencies()
    opencv_ok = check_opencv_pose_support()
    
    if not success:
        print("\n🔧 Manual installation:")
        print("pip install --upgrade ultralytics")
        show_alternative_methods()
        sys.exit(1)
    else:
        print(f"\n✅ Setup complete! OpenCV DNN: {'Available' if opencv_ok else 'Not available'}")
        show_alternative_methods()