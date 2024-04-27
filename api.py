from fastapi import FastAPI, File, UploadFile
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import io
import google.generativeai as genai
from deep_translator import GoogleTranslator

app = FastAPI(debug=True)

# Load the trained model
model_path = 'model/corn_leaf_disease.pth'

classes_dict = {
    "corn_cercospora_leaf_spot_gray_leaf_spot": "Corn Cercospora Leaf Spot/Gray Leaf Spot",
    "corn_common_rust": "Corn Common Rust",
    "corn_healthy": "Corn Healthy",
    "corn_northern_leaf_blight": "Corn Northern Leaf Blight"
}

# Load the model architecture
class MobileNetV3(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

@app.post("/predict/")
async def image_predict(image: UploadFile = File(...)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model = MobileNetV3(150528).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the expected size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor
    ])
    
    # Read the image file
    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data))
    image_tensor = transform(pil_image)
    
    # Unsqueeze to add batch dimension
    image_tensor = image_tensor.unsqueeze(0) # type: ignore
    image_tensor = image_tensor.to(device)
    
    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    # Assuming you have the data_class object
    data_class = ImageFolder(root="corn_leaf_classification/training")
    predicted_class_label = data_class.classes[predicted_class.item()] # type: ignore
    predicted_class_probability = probabilities[0][predicted_class].item()
    
    # Konfigurasi kunci API
    genai.configure(api_key='AIzaSyDpEh8S4jo__bjNtJy2hN9cX838FZyF4Ww')

    # Inisialisasi model Gemini Pro
    model_ai_gemini = genai.GenerativeModel('gemini-pro')
    
    if predicted_class_label != "corn_healthy":
        query =f"""
        apa itu {predicted_class_label}?
        kenapa bisa gitu? dan bagaimana cara mengatasinya?
        """
        
        response = model_ai_gemini.generate_content(query)
         
    if predicted_class_label == "corn_healthy":
        query = f"""
        
        berikan tips kepada pengguna untuk cara merawat tanamannya.
        
        """
        
        response = model_ai_gemini.generate_content(query)
        # translated = GoogleTranslator(source='auto', target='de').translate("keep it up, you are awesome")
    
    return {
        "class": predicted_class_label,
        "probability": predicted_class_probability * 100,
        "response": response.text
    }
    