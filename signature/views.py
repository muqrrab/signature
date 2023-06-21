from django.shortcuts import render, redirect
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt
import json
import cv2
import numpy as np
import requests
from io import BytesIO
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as kimage

def download_image(url):
    response = requests.get(url)
    image = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
model = VGG16(weights='imagenet', include_top=False)
def extract_features(image):
    global model
    try:
        image = cv2.resize(image, (224, 224))
        img_array = kimage.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        features = features.flatten()
        return features
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")

def normalize_features(features):
    features = (features - np.mean(features)) / np.std(features)
    return features

def verify_signature(input_signature, reference_signature):
    input_signature = normalize_features(input_signature)
    reference_signature = normalize_features(reference_signature)
    distance = cosine(input_signature, reference_signature)
    similarity = 1 - distance
    percentage_similarity = round(similarity * 100, 2)
    return percentage_similarity

@csrf_exempt
def verify_signature_route(request):
    # data = request.FILES['file']
    data = json.loads(request.body)
    if not data:
        return JsonResponse({"message": "No input data provided"}, safe=False)

    reference_signature_url = data['reference_signature_url']
    original_signature_urls = data['original_signature_urls']

    if not reference_signature_url or not original_signature_urls:
        return JsonResponse({"message": "Invalid input data"}, safe=False)

    try:
        reference_signature_image = download_image(reference_signature_url)
        print('inside')
        reference_signature_features = extract_features(reference_signature_image)
        results = []

        # threshold = 0.95

        for original_signature in original_signature_urls:
            user_id = original_signature.get('userid')
            designation = original_signature.get('designation')
            full_name = original_signature.get('fullname')
            signature_url = original_signature.get('signature')
            image_url = original_signature.get('image_url')

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
                    'reference_signature_url': reference_signature_url,
                    'image_url': image_url
                }
                results.append(result)

        if results:
            sorted_results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
            # if sorted_results[0]:
            result = sorted_results[0]
            return JsonResponse(result, safe=False)
            # else:
            #     return JsonResponse({"message": "No matching signatures found"}, safe=False)
        else:
            return JsonResponse({"message": "No matching signatures found"}, safe=False)

    except ValueError as e:
        return JsonResponse({"message": str(e)}, safe=False)


@csrf_exempt
def index(request):
    return JsonResponse('Hello, world', safe=False)