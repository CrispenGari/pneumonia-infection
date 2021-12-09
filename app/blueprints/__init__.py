
from flask import Blueprint, make_response, jsonify, request
from PIL import Image
from model import make_prediction, model, preprocess_img
import io

blueprint = Blueprint("blueprint",__name__)

@blueprint.route('/pneumonia', methods=["POST"])
def classify_pneumonia():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files.get("image").read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            image = preprocess_img(image)
            preds = make_prediction(model, image)
            
            data["success"] = True
            data["predictions"] = preds     
    return make_response(jsonify(data)), 200