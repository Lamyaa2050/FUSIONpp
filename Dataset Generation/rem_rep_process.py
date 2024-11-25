from essentials import *
from packages import *

def inpaint_with_lama(image_path, mask_path, output_path, feather_radius=2, blur_kernel=0.1, median_kernel=1, upscale_factor=1):
    try:
        print(f"[INFO] 🖼️ Inpainting {image_path} with mask {mask_path}")
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert("L")
        if image.size != mask.size:
            raise ValueError("Image and mask must have the same dimensions.")
        if feather_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        if upscale_factor > 1:
            image = image.resize(
                (int(image.width * upscale_factor), int(image.height * upscale_factor)),
                Image.LANCZOS,
            )
            mask = mask.resize(image.size, Image.LANCZOS)
        simple_lama = SimpleLama()
        result = simple_lama(image, mask)
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result = cv2.medianBlur(result, median_kernel)
        if upscale_factor > 1:
            result = cv2.resize(result, (image.width, image.height), interpolation=cv2.INTER_LANCZOS4)
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        result.save(output_path)
        print(f"[INFO] ✅ Inpainted image saved to: {output_path}")
        return image, mask, result
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"[ERROR] ❌ Error: {e}")
        return None, None, None

def plot_images(original_image, mask, inpainted_image, image_name):
    original_image_np = np.array(original_image)
    mask_np = np.array(mask)
    inpainted_image_np = np.array(inpainted_image)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Mask')
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Part 1 : Removement')
    plt.imshow(inpainted_image_np)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def process_images(image_folder, mask_folder, output_folder, batch_size=5):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    total_images = len(image_files)
    print(f"[INFO] Total images to process: {total_images}")

    for i in range(0, total_images, batch_size):
        print(f"[INFO] Processing batch {i//batch_size + 1}")
        batch_files = image_files[i:i + batch_size]
        for image_file in batch_files:
            image_name = os.path.splitext(image_file)[0]
            output_path = os.path.join(output_folder, f"{image_name}.png")
            if os.path.exists(output_path):
                print(f"[INFO] ⏩ Skipping {output_path} (already exists)")
                continue
            corresponding_mask = f"{image_name}mask0.png"
            if corresponding_mask in os.listdir(mask_folder):
                image_path = os.path.join(image_folder, image_file)
                mask_path = os.path.join(mask_folder, corresponding_mask)
                original_image, mask, inpainted_image = inpaint_with_lama(image_path, mask_path, output_path, upscale_factor=1)
                if original_image and mask and inpainted_image:
                    plot_images(original_image, mask, inpainted_image, image_name)
                gc.collect()
            else:
                print(f"[ERROR] No corresponding mask found for {image_file}")
        gc.collect()
        print(f"[INFO] Finished processing batch {i//batch_size + 1}")
