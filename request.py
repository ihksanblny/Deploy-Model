import requests
import json

def check_system():
    # Base URL
    base_url = 'http://localhost:5000'
    
    # 1. Check if server is running
    try:
        # Check experts data
        print("\nChecking experts data...")
        response = requests.get(f'{base_url}/check-experts')
        if response.status_code == 200:
            print("Experts data response:", json.dumps(response.json(), indent=2))
        else:
            print(f"Error checking experts: {response.status_code}")
            print("Response text:", response.text)
            
        # Test image processing
        print("\nTesting image processing...")
        files = {'image': open('./static/input_1.jpg', 'rb')}
        response = requests.post(f'{base_url}/process-question', files=files)
        if response.status_code == 200:
            print("Process result:", json.dumps(response.json(), indent=2))
        else:
            print(f"Error processing image: {response.status_code}")
            print("Response text:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running?")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_system()