from essentials import *
from packages import *

def convert_jpg_to_png(image_path, save_path):
    img = Image.open(image_path)
    img.save(save_path)
    print(f"Converted JPEG image to PNG: {save_path}")

def process_images_in_folder(data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(data_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            input_path = os.path.join(data_folder, filename)
            output_filename = filename[:-4] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            convert_jpg_to_png(input_path, output_path)