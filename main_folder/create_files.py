import os

def create_image_lists(input_folder, output_base_folder, overlap=10):
    for folder_name in os.listdir(input_folder):
        image_folder = os.path.join(input_folder, folder_name)
        output_folder = os.path.join(output_base_folder, folder_name)
        
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        image_files.sort()

        num_images = len(image_files)
        num_lists = num_images // (100 - overlap)

        os.makedirs(output_folder, exist_ok=True)

        for i in range(num_lists):
            start_index = i * (100 - overlap)
            if i == (num_lists - 1):
                end_index = num_images
            else:
                end_index = min((i + 1) * (100 - overlap) + overlap, num_images)
            list_name = os.path.join(output_folder, f"list{i}.txt")

            with open(list_name, 'w') as f:
                for image_file in image_files[start_index:end_index]:
                    f.write(image_file + '\n')

            print(f"Created {list_name} with {end_index - start_index} images.")

# Directorio que contiene las carpetas con imágenes
input_base_folder = "/home/david_serna/Documents/SfM_Kiwibot_Repository/SfM_CUDA/colmap_ws/rosbag_office/images"
# Directorio base de salida para los datos generados
output_base_folder = "/home/david_serna/Documents/SfM_Kiwibot_Repository/SfM_CUDA/colmap_ws/rosbag_office/lists_folder"

create_image_lists(input_base_folder, output_base_folder)
