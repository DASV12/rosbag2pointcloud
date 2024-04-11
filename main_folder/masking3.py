import os
import cv2
import torch
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-seg.pt')

# Directorio de entrada y salida
input_folder = '/home/david_serna/Documents/SfM_Kiwibot_Repository/SfM/colmap_ws/rosbag_office/images/right/'
output_folder = '/home/david_serna/Documents/SfM_Kiwibot_Repository/SfM/colmap_ws/rosbag_office/masks/right/'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Obtener la lista de archivos en el directorio de entrada y ordenarlos alfabéticamente
files = os.listdir(input_folder)
files.sort()
print(files)
input("Presiona Enter para continuar...")

# Iterar sobre todas las imágenes en el directorio de entrada
for filename in files:
    if filename.endswith('.jpg'):
        # Cargar la imagen original
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Ejecutar la segmentación en la imagen original
        results = model.predict(image, save=False, imgsz=[736,1280])
        
        # Inicializar la máscara combinada
        image_shape = (image.shape[0], image.shape[1])  # Tamaño de la imagen original
        combined_mask = torch.zeros([736,1280], dtype=torch.uint8)
        
        # Iterar sobre los resultados
        for result in results:
            # Verificar si se detectaron máscaras en el resultado
            if result.masks is not None:
                # Extraer las máscaras y las cajas de detección
                masks = result.masks.data
                boxes = result.boxes.data
                
                # Resto del código para procesar las máscaras
            else:
                print("No se detectaron máscaras en la imagen.")
                continue
            
            # Extraer las clases
            clss = boxes[:, 5]
            
            # Obtener índices de las clases de interés (personas, carros, camiones y motocicletas)
            indices_personas = torch.where(clss == 0)[0]
            indices_carros = torch.where(clss == 2)[0]
            indices_camiones = torch.where(clss == 7)[0]
            indices_motocicletas = torch.where(clss == 3)[0]

            # Combinar las máscaras de las clases de interés en una sola máscara para cada clase
            people_mask = torch.any(masks[indices_personas], dim=0).int() * 255
            car_mask = torch.any(masks[indices_carros], dim=0).int() * 255
            truck_mask = torch.any(masks[indices_camiones], dim=0).int() * 255
            motorcycle_mask = torch.any(masks[indices_motocicletas], dim=0).int() * 255

            # Sumar las máscaras combinadas a la máscara combinada general
            combined_mask += people_mask + car_mask + truck_mask + motorcycle_mask
        
        # Escalar la máscara combinada para que coincida con las dimensiones de la imagen original
        combined_resized_mask = cv2.resize(combined_mask.cpu().numpy(), (image.shape[1], image.shape[0]))
        
        # Invertir los colores de la máscara combinada
        combined_inverted_mask = cv2.bitwise_not(combined_resized_mask)
        
        # Guardar la máscara combinada invertida con el mismo nombre que la imagen original
        output_path = os.path.join(output_folder, filename.replace('jpg', 'jpg.png'))
        cv2.imwrite(output_path, combined_inverted_mask)
