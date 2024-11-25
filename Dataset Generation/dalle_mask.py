from packages import *

def make_black_transparent(image_path, save_path=None):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    data = img.getdata()
    new_data = []
    for item in data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    if save_path is not None:
        if os.path.isdir(save_path):
            filename = os.path.basename(image_path)
            save_path = os.path.join(save_path, filename)
        else:
            directory = os.path.dirname(save_path)
            os.makedirs(directory, exist_ok=True)
    else:
        save_path = image_path
    img.save(save_path)

def process_images_in_folder(temp_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(temp_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(temp_folder, filename)
            transparent_filename = filename[:-4] + "_transparent.png"
            save_path = os.path.join(output_folder, transparent_filename)
            make_black_transparent(image_path, save_path)
            print(f"Processed {filename} and saved as {transparent_filename}")