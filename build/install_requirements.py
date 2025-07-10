import subprocess
import sys
import os

def install_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.isfile(req_path):
        print("requirements.txt not found.")
        sys.exit(1)

    print("Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("\n✅ Installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
