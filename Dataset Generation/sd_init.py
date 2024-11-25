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

class ImageDataset(Dataset):
    def __init__(self, image_dir, max_size=512):
        if not os.path.exists(image_dir):
            raise ValueError(f"The directory {image_dir} does not exist.")
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.max_size = max_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_and_pad(image)
        image = image.astype(np.float32) / 255.0
        return image, self.image_filenames[idx]

    def resize_and_pad(self, image):
        h, w = image.shape[:2]
        scale = self.max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        new_h, new_w = image.shape[:2]
        top = (self.max_size - new_h) // 2
        bottom = self.max_size - new_h - top
        left = (self.max_size - new_w) // 2
        right = self.max_size - new_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image

def yolov10_detection(model, image):
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(model.device)
    image = image.float()
    image /= 255.0
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    return boxes, labels

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def process_image(image, image_name, yolov10_model, predictor, threshold_area, output_mask_dir, output_temp_dir, output_crop_dir):
    def attempt_detection_and_segmentation(image, attempt=0):
        yolov10_boxes, label_indices = yolov10_detection(yolov10_model, image)
        predictor.set_image(image)
        filtered_masks = []
        for box in yolov10_boxes:
            input_box = np.array(box)
            masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=True)
            masks_with_areas = [(mask, np.sum(mask)) for mask in masks]
            for mask, area in masks_with_areas:
                if area > threshold_area:
                    if not any(np.array_equal(mask, fm[0]) for fm in filtered_masks):
                        filtered_masks.append((mask, area, input_box))

        return filtered_masks

    if os.path.exists(os.path.join(output_mask_dir, f"{os.path.splitext(image_name)[0]}_mask_1.png")):
        print(f"\n🔄 [INFO] Image '{image_name}' already processed. Skipping.")
        return
    image = (image * 255).astype(np.uint8)
    filtered_masks = attempt_detection_and_segmentation(image)
    attempts = 0
    while not filtered_masks and attempts < 3:
        print(f"[INFO] No masks found for '{image_name}' on attempt {attempts + 1}. Retrying...")
        attempts += 1
        filtered_masks = attempt_detection_and_segmentation(image, attempt=attempts)
    if not filtered_masks:
        print(f"[INFO] No masks found for '{image_name}' after {attempts} attempts. Using bounding boxes as fallback.")
        yolov10_boxes, _ = yolov10_detection(yolov10_model, image)
        h, w = image.shape[:2]
        for box in yolov10_boxes:
            mask = np.zeros((h, w), dtype=np.uint8)
            x0, y0, x1, y1 = map(int, box)
            mask[y0:y1, x0:x1] = 1
            filtered_masks.append((mask, np.sum(mask), box))

    if filtered_masks:
        mask, area, box = max(filtered_masks, key=lambda x: x[1])
        binary_mask = torch.from_numpy(mask).squeeze().numpy().astype(np.uint8)
        mask_image = Image.fromarray(binary_mask * 255)
        mask_name = os.path.splitext(image_name)[0] + "_mask.png"
        mask_output_path = os.path.join(output_mask_dir, mask_name)
        mask_image.save(mask_output_path)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Warning: No contours found for the largest mask in image {image_name}. Skipping...")
            return
        largest_contour = max(contours, key=cv2.contourArea)
        bbox = [int(x) for x in cv2.boundingRect(largest_contour)]
        cropped_image = image.copy()
        cropped_image[binary_mask == 0] = 0
        crop_name = os.path.splitext(image_name)[0] + "_cropped.png"
        crop_output_path = os.path.join(output_crop_dir, crop_name)
        cv2.imwrite(crop_output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        blended_image = image.copy()
        blended_image[binary_mask == 1] = 0
        temp_name = os.path.splitext(image_name)[0] + "_blended.png"
        temp_output_path = os.path.join(output_temp_dir, temp_name)
        cv2.imwrite(temp_output_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_box(box, plt.gca())
        plt.axis('off')
        plt.show()
        print(f"Image: {image_name}")
        print("Bounding box:", bbox)
        print("Segmentation mask:", largest_contour.flatten().tolist())

def process_images_with_dataloader(input_dir, output_mask_dir, output_temp_dir, output_crop_dir, batch_size=10):
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_temp_dir, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)
    yolov10_model = YOLO('yolov10x.pt').to('cuda')
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    dataset = ImageDataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for images, names in tqdm(dataloader, desc="Processing Batches"):
        for image, name in zip(images, names):
            process_image(image.numpy(), name, yolov10_model, predictor, 1000, output_mask_dir, output_temp_dir, output_crop_dir)
            gc.collect()


