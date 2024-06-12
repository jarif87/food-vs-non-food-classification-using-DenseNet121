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
        # Convert image to PIL Image object if it's not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert image to RGB mode if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess image
        image_size = (224, 224)
        image_resized = image.resize(image_size)
        image_np = np.array(image_resized) / 255.0
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Make prediction
        prediction = model.predict(image_np_expanded)
        final_prediction = np.argmax(prediction[0])

        # Display result
        results = {0: 'Food', 1: 'Non Food'}
        label = results[final_prediction]
        
        # Create a draw object
        draw = ImageDraw.Draw(image)
        
        # Determine label color and font size based on prediction
        if final_prediction == 0:
            label_color = (255, 0, 0)  # Red for "Food"
            text_font = ImageFont.truetype("Hack-Regular.ttf", 24)  # Font size for "Food"
            text_bbox = draw.textbbox((0, 0), label, font=text_font)
            text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
            text_position = ((image_size[0] - text_size[0]) // 2, 10)
        else:
            label_color = (0, 255, 0)  # Green for "Non Food"
            text_font = ImageFont.truetype("Hack-Regular.ttf", 48)  # Increased font size for "Non Food"
            text_bbox = draw.textbbox((0, 0), label, font=text_font)
            text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
            text_position = ((image_size[0] - text_size[0]) // 2, (image_size[1] - text_size[1]) // 2)
        
        # Add text to the image
        draw.text(text_position, label, fill=label_color, font=text_font)

        # Return modified image as a PIL Image object
        return image
    except Exception as e:
        print("Error processing image:", e)
        # Return a blank white image
        return Image.new("RGB", (224, 224), (255, 255, 255))

# Define input component
image_input = "image"

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
