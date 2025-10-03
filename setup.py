import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        'transformers',
        'sentence-transformers', 
        'faiss-cpu',
        'streamlit',
        'pandas',
        'numpy',
        'torch',
        'pickle5'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def setup_project():
    """Setup the project structure"""
    print("Setting up MeitY Chatbot...")
    
    # Install packages
    install_requirements()
    
    # Check if data file exists
    if not os.path.exists('cleaned.json'):
        print("\nPlease add your JSON data file as 'cleaned.json'")
        print("Then run: python process_data.py")
    else:
        print("\n✓ Data file found")
        print("Run: python process_data.py")
    
    print("\nAfter processing data, run: streamlit run app.py")

if __name__ == "__main__":
    setup_project()