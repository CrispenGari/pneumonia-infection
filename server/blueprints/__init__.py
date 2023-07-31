from flask import Blueprint, make_response, jsonify, request
from PIL import Image
from models import v0, v1
import io

blueprint = Blueprint("blueprint", __name__)


@blueprint.route("v0/pneumonia", methods=["POST"])
def classify_pneumonia_v0():
    data = {
        "success": False,
        "modelVersion": "v0",
        "meta": {
            "programmer": "@crispengari",
            "main": "computer vision (cv)",
            "description": "given a medical chest-x-ray image of a human being we are going to classify weather a person have pneumonia virus, pneumonia bacteria or none of those(normal).",
            "language": "python",
            "library": "pytorch",
        },
    }
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files.get("image").read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            image = v0.preprocess_img(image)
            preds = v0.make_prediction(v0.pneumonia_mpl, image)

            data["success"] = True
            data["predictions"] = preds
    return make_response(jsonify(data)), 200


@blueprint.route("v1/pneumonia", methods=["POST"])
def classify_pneumonia_v1():
    data = {
        "success": False,
        "modelVersion": "v1",
        "meta": {
            "programmer": "@crispengari",
            "main": "computer vision (cv)",
            "description": "given a medical chest-x-ray image of a human being we are going to classify weather a person have pneumonia virus, pneumonia bacteria or none of those(normal).",
            "language": "python",
            "library": "pytorch",
        },
    }
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files.get("image").read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            image = v1.preprocess_img(image)
            preds = v1.make_prediction(v1.pneumonia_lenet, image, v1.device)
            data["success"] = True
            data["predictions"] = preds
    return make_response(jsonify(data)), 200
