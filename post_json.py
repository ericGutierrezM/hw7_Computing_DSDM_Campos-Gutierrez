import requests

url = "http://localhost:8000/predict-file"

try:
    with open("example_input.json", "rb") as f:
        response = requests.post(url, files={"file": f}, timeout=20)
    response.raise_for_status()
    print("Prediction:", response.json()["prediction"])

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print("Error 400: Bad Request, check the JSON input format or data types.")
    elif e.response.status_code == 404:
        print("Error 404: Not Found, the endpoint does not exist.")
    elif e.response.status_code == 500:
        print("Error 500: Internal Server Error, something went wrong on the server.")
    elif e.response.status_code == 503:
        print("Error 503: Service Unavailable, the server is overloaded or down.")
    else:
        print(f"HTTP error occurred: {e.response.status_code}")
    print("Response content:", e.response.text)