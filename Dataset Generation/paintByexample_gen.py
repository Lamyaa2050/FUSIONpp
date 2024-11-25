from essentials import *
from packages import *

def limit_descriptions(descriptions, max_objects=3):
    return descriptions[:max_objects]

def resize_image(image, target_size):
    return image.resize(target_size, Image.ANTIALIAS)

def get_image_prompt(image_path, crop_paths, processor, model):
    image = Image.open(image_path).convert('RGB')
    descriptions = []
    print(f"[DEBUG] Image Path: {image_path}")
    print(f"[DEBUG] Crop Paths: {crop_paths}")
    for crop_path in crop_paths:
        try:
            if os.path.exists(crop_path):
                crop = Image.open(crop_path).convert('RGB')
                print(f"[INFO] 📸 Processing crop image: {crop_path}")
                masked_image_inputs = processor(images=crop, return_tensors="pt")
                masked_image_out = model.generate(**masked_image_inputs)
                masked_object_description = processor.batch_decode(masked_image_out, skip_special_tokens=True)[0]
                descriptions.append(masked_object_description)
                print(f"[INFO] 🖼️ Crop Description: {masked_object_description}")
            else:
                print(f"[WARNING] ⚠️ Crop path {crop_path} does not exist.")
                descriptions.append("")
        except Exception as e:
            print(f"[ERROR] ❌ Error processing crop path {crop_path}: {e}")
            descriptions.append("")
    image_inputs = processor(images=image, return_tensors="pt")
    image_out = model.generate(**image_inputs)
    image_description = processor.batch_decode(image_out, skip_special_tokens=True)[0]
    print(f"[INFO] 📸 Image Description: {image_description}")
    print(f"[INFO] 🖼️ Mask Descriptions (Crop Paths): {descriptions}")
    user_message_parts = [f"remove the {desc}" for desc in descriptions if desc]
    user_message = f"Generate a prompt to {' and '.join(user_message_parts)} in an image similar to '{image_description}', focusing on the areas defined by the provided masks. Ensure the objects are removed seamlessly and the remaining scene looks natural."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are skilled in creating prompts for DALL-E 2 image editing."},
            {"role": "user", "content": user_message}
        ]
    )
    return completion.choices[0].message.content.strip(), image_description, descriptions


