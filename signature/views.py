from django.shortcuts import render, redirect
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt

import cv2
import numpy as np
import requests
from scipy.spatial.distance import cosine
from keras.applications import ResNet50

import json

model = ResNet50(weights='imagenet', include_top=False)

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises stored HTTPError, if one occurred.
    except requests.HTTPError as http_err:
        raise ValueError(f'Error downloading image: {http_err}') 
    except Exception as err:
        raise ValueError(f'Error downloading image: {err}')
    else:
        image = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours were found, return None
    if not contours:
        return None

    # Otherwise, find largest contour
    c = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop the image to this bounding rectangle
    return gray[y:y+h, x:x+w]

def extract_features(image):
    image = preprocess_image(image)
    if image is None:
        return None

    image = cv2.resize(image, (224, 224)) # ResNet50 expects input size of (224, 224)
    image = image / 255.0 # Normalize pixel values
    image = np.stack((image,)*3, axis=-1) # Adding an extra dimension for channels
    image = np.expand_dims(image, axis=0) # Expand dimensions for model prediction

    features = model.predict(image)
    return features.flatten()


def verify_signature(input_signature, reference_signature, threshold=0.95):
    if input_signature is not None and reference_signature is not None:
        distance = cosine(input_signature, reference_signature)
        similarity = 1 - distance

        if similarity >= threshold:
            return similarity
        else:
            return 0.0  # Return 0 for non-matching signatures
    else:
        return None

@csrf_exempt
def verify_signature_route(request):
    # data = request.FILES['file']
    data = json.loads(request.body)
    if not data:
        return JsonResponse({"message": "No input data provided"}, safe=False)

    print(data)
    reference_signature_url = data['reference_signature_url']
    original_signature_urls = data['original_signature_urls']

    if not reference_signature_url or not original_signature_urls:
        return JsonResponse({"message": "Invalid input data"}, safe=False)

    try:
        reference_signature_image = download_image(reference_signature_url)
        reference_signature_features = extract_features(reference_signature_image)
        results = []

        threshold = 0.95

        for original_signature in original_signature_urls:
            user_id = original_signature.get('userid')
            designation = original_signature.get('designation')
            full_name = original_signature.get('fullname')
            signature_url = original_signature.get('signature')

            original_signature_image = download_image(signature_url)
            original_signature_features = extract_features(original_signature_image)
            similarity = verify_signature(original_signature_features, reference_signature_features)

            if similarity is not None and similarity > 0.0:
                result = {
                    'user_id': user_id,
                    'designation': designation,
                    'full_name': full_name,
                    'signature_url': signature_url,
                    'similarity_score': similarity,
                    'reference_signature_url': reference_signature_url
                }
                results.append(result)

        if results:
            sorted_results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
            if sorted_results[0]['similarity_score'] >= threshold:
                result = sorted_results[0]
                return JsonResponse(result, safe=False)
            else:
                return JsonResponse({"message": "No matching signatures found"}, safe=False)
        else:
            return JsonResponse({"message": "No matching signatures found"}, safe=False)

    except ValueError as e:
        return JsonResponse({"message": str(e)}, safe=False)


@csrf_exempt
def index(request):
    return JsonResponse('Hello, world', safe=False)