import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
from model import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

# instantiate a parser
parser = argparse.ArgumentParser()
# add an argument '--model_path'
parser.add_argument('--model_path', type=str, default = 'models/mnist_net.pth', help='model path')
args = parser.parse_args()
model_path = args.model_path

model = MNISTNet().to(device)
# Load the model
model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": int(predicted[0])})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    imgs_binary = request.files.getlist("images[]")
    

    # Transform the PIL image
    tensors = []
    for img_binary in imgs_binary:
        img_pil = Image.open(img_binary.stream)
        tensor = transform(img_pil).to(device)
        tensors.append(tensor)

    # make batch tensor
    tensors = torch.stack(tensors, dim=0)

    # Make prediction
    with torch.no_grad():
        outputs = model(tensors)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": predicted.tolist()})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)