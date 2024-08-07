import urllib.request
import certifi
import ssl

# URL to test
url = "https://tfhub.dev/google/facenet/1"

# Create an SSL context using certifi's CA bundle
context = ssl.create_default_context(cafile=certifi.where())

try:
    # Make a request to the URL using the custom SSL context
    response = urllib.request.urlopen(url, context=context)
    print("SSL certificate verification succeeded.")
    print("Response status code:", response.getcode())
    print("Response headers:", response.info())
except Exception as e:
    print(f"SSL certificate verification failed: {e}")

