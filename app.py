import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Function to load the modified model without recompiling
def load_modified_model(model_path):
    return tf.keras.models.load_model(model_path)

# Load the trained model
print("Loading model...")
model = load_modified_model('food_model_1.h5')
print("Model loaded successfully.")

# Function to classify food vs. non-food image using the loaded model
def classify_food_vs_nonfood(image):
    try:
        # Preprocess image
        image_size = (224, 224)
        image = image.resize(image_size)
        image_np = np.array(image) / 255.0
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Make prediction
        prediction = model.predict(image_np_expanded)
        final_prediction = np.argmax(prediction[0])

        # Display result
        results = {0: 'Food', 1: 'Non Food'}
        label = results[final_prediction]
        
        # Create a draw object
        draw = ImageDraw.Draw(image)
        
        # Specify font and size
        font = ImageFont.load_default()
        
        # Get text size
        text_font = ImageFont.truetype("Hack-Regular.ttf", 24)
        text_bbox = draw.textbbox((0, 0), label, font=text_font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        
        # Calculate text position
        text_position = ((image_size[0] - text_size[0]) // 2, 10)
        
        # Add text to the image
        draw.text(text_position, label, fill=(255, 0, 0), font=text_font)

        # Return modified image
        return image
    except Exception as e:
        print("Error processing image:", e)

# Define inputs for Gradio interface
image_input = gr.inputs.Image(shape=(224, 224), type="pil")

# Define example images as file paths
ex_image_paths = ['image_1.jpeg', 'image_2.jpeg', 'image_3.jpeg', 'image_4.jpg', 'image_5.jpg']

# Launch Gradio interface with example images
food_vs_nonfood_interface = gr.Interface(classify_food_vs_nonfood, 
                                         inputs=image_input, 
                                         outputs="image",
                                         title="Food vs NonFood Classifier",
                                         description="Upload an image to classify whether it's food or non-food.",
                                         examples=ex_image_paths)
food_vs_nonfood_interface.launch(inline=False)