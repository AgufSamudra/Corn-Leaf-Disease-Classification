import streamlit as st
from PIL import Image
import requests

# Define the URL of your FastAPI endpoint
fastapi_url = "http://127.0.0.1:8000/predict/"

classes_dict = {
    "corn_cercospora_leaf_spot_gray_leaf_spot": "Corn Cercospora Leaf Spot/Gray Leaf Spot",
    "corn_common_rust": "Corn Common Rust",
    "corn_healthy": "Corn Healthy",
    "corn_northern_leaf_blight": "Corn Northern Leaf Blight"
}


def ui_interface():
    st.header("Corn Leaf Classification", divider="rainbow")
    upload_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if upload_image:
        image = Image.open(upload_image).convert("RGB")
        st.image(image)
        
        # Button for prediction
        button_predict = st.button("Predict", type="primary", use_container_width=True)
        if button_predict:
            img_bytes = upload_image.getvalue() 
            
            response = requests.post(
                fastapi_url,
                files={"image": ("filename.jpg", img_bytes, "image/jpeg")}
            )
            
            if response.status_code == 200:
                prediction_result = response.json()
                # st.write(prediction_result)
                st.markdown(f"""
# Prediction

##### Class: {classes_dict[prediction_result["class"]]}
##### Probabilitas: {prediction_result["probability"]:.2f}%
##### Response:<br> {prediction_result["response"]}
""", unsafe_allow_html=True)
                
            else:
                st.error(f"Failed to get prediction. HTTP Status Code: {response.status_code}")
    else:
        pass

if __name__ == "__main__":
    ui_interface()
