import subprocess
import sys
import time

def run_services():
    # Start the FastAPI server
    api_process = subprocess.Popen([sys.executable, "api.py"])
    print("Started FastAPI server...")
    
    # Wait for the API server to start
    time.sleep(5)
    
    # Start the Streamlit client
    streamlit_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_api_client.py"])
    print("Started Streamlit client...")
    
    try:
        # Keep the script running
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("\nShutting down services...")
        api_process.terminate()
        streamlit_process.terminate()
        api_process.wait()
        streamlit_process.wait()

if __name__ == "__main__":
    run_services() 