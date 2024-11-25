from essentials import *
from packages import *
from rem_rep_init import *
from rem_rep_process import *
from rem_rep_comb import *
from rem_rep_main import *

data_folder = "data_rem_rep"
output_folder = "data_rem_rep_png"
process_images_in_folder_(data_folder, output_folder)

input_dir = "data_rem_rep_png"
output_mask_dir = "mask_temp_1"
output_temp_dir = "temp_temp_1"
output_crop_dir = "crop_temp_1"
threshold_area = 10000
csv_file_path = "output_temp_1.csv"
process_images_in_folder(input_dir, 
                         output_mask_dir, 
                         output_temp_dir, 
                         output_crop_dir, 
                         threshold_area, 
                         csv_file_path)

image_folder = "data_rem_rep_png"
mask_folder = "mask_temp_1"
output_folder = "output_temp_1"
process_images(image_folder, mask_folder, output_folder)

input_dir = os.path.join(HOME, "output_temp_1")
output_mask_dir = os.path.join(HOME, "mask_temp_2")
output_temp_dir = os.path.join(HOME, "temp_temp_2")
output_crop_dir = os.path.join(HOME, "crop_temp_2")
csv_file_path = os.path.join(HOME, "mask_labels_temp_2.csv")
threshold_area = 5000
process_images_in_folder(input_dir, 
                         output_mask_dir, 
                         output_temp_dir, 
                         output_crop_dir, 
                         threshold_area, 
                         csv_file_path)

temp_folder = "temp_temp_2"
output_folder = "transparent_temp"
process_images_in_folder__(temp_folder, output_folder)


folder1 = 'mask_temp_1'
folder2 = 'mask_temp_2'
output_folder = 'mask_rem_rep'

os.makedirs(output_folder, exist_ok=True)
filenames = os.listdir(folder1)

for filename in filenames:
    if filename.endswith(".png") or filename.endswith(".jpg"):
        mask1 = read_mask(folder1, filename)
        mask2 = read_mask(folder2, filename)

        if mask1 is not None and mask2 is not None:
            merged_mask = merge_masks(mask1, mask2)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, merged_mask)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.title('Mask 1')
            plt.imshow(mask1, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('Mask 2')
            plt.imshow(mask2, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Merged Mask')
            plt.imshow(merged_mask, cmap='gray')
            plt.axis('off')

            plt.show()

data_png_folder = "output_temp_1"
transparent_folder = "transparent_temp"
crop_folder = "crop_temp_2"
output_dalle_creation_folder = "output_rem_rep"
mask_rem_rep_folder = "mask_rem_rep"
data_rem_rep_png_folder = "data_rem_rep_png"
LEVELS = ["replacement"]
process_folders_(LEVELS, 
                 data_png_folder, 
                 transparent_folder, 
                 crop_folder, 
                 output_dalle_creation_folder, 
                 mask_rem_rep_folder, 
                 data_rem_rep_png_folder, 
                 des_processor, 
                 des_model)
