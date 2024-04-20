import os
import typing as tp
import cv2
import numpy as np
import message_filters
import rosbag2_py
import sensor_msgs
import tf2_msgs
import geometry_msgs
#from tf2_msgs.msg import TFMessage
import typer
import piexif
import matplotlib.pyplot as plt
import sys
import torch
from ultralytics import YOLO
#import transforms3d.quaternions as quat
#import tf.transformations as tf
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tqdm import tqdm
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

import zstandard


def decompress_file(input_path: str, output_path: str):
    """! Decompress a file using zstandard
    @param input_path (str) path to the compressed file
    @param output_path (str) path to the decompressed file
    """
    dctx = zstandard.ZstdDecompressor()

    with open(input_path, "rb") as ifh, open(output_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)


# Avoid printing matrices with scientific notation
np.set_printoptions(suppress=True)


def get_rosbag_options(path: str, serialization_format="cdr"):
    """! Ros2 bag options
    @param path (str) path to rosbag2
    @param serialization_format (str, optional) format . Defaults to "cdr".
    @return tuple of storage and converter options
    """
    extension_file = path.split(".")[-1]
    if extension_file == "mcap":
        storage_id = "mcap"
    elif extension_file == "db3":
        storage_id = "sqlite3"
    else:
        raise ValueError(f"Unknown extension file {extension_file}")
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )
    return (storage_options, converter_options)


class RosBagSerializer(object):
    def __init__(
        self,
        rosbag_path: str,
        topics: tp.List[str],
        queue_size: int = 30,
        time_delta: float = 0.1,
        imshow: bool = True,
        undistort: bool = True,
        fps: int = 10,
        verbose=True,
    ):
        """! Serialize a rosbag into different synchronized videos per topic.
        Optionally, it can also save a video with all the topics concatenated.

        @param rosbag_path (str) path to the rosbag to serialize.
        @param topics (list) list of topics to serialize and synchronize.
        @param queue_size (int, optional) queue size for the message filters. Defaults to 30.
        @param time_delta (float, optional) time delta for the message filters. Defaults to 0.1.
        @param imshow (bool, optional) show the images while serializing. Defaults to False.
        @param undistort (bool, optional) undistort the images using the camera info. Defaults to True.
        @param concat_images (bool, optional) concatenate all the images in a single video. Defaults to False.
        @param fps (int, optional) fps for the videos. Defaults to 10.
        @param verbose (bool, optional) print information while serializing. Defaults to True.
        """

        self.rosbag_path = rosbag_path
        self.topics = topics
        self.original_topics = topics
        self.queue_size = queue_size
        self.time_delta = time_delta
        self.imshow = imshow
        self.fps = fps
        self.undistort = undistort
        self.verbose = verbose

        self.rosbag_dir = os.path.dirname(rosbag_path)
        self.bridge = CvBridge()
        # Crear listas para almacenar los valores de pose_x y pose_y en cada ciclo y graficar
        self.pose_x_values = []
        self.pose_y_values = []
        self.is_gps = bool


        self.output_dir = os.path.join(os.path.expanduser('/working/colmap_ws'), 'images')
        os.makedirs(self.output_dir, exist_ok=True)

        self.imu_gps_output_dir = os.path.join(os.path.expanduser('/working'), 'colmap_ws')
        os.makedirs(self.imu_gps_output_dir, exist_ok=True)

        tf_data_file = os.path.join(self.imu_gps_output_dir, 'tf_data.txt')
        gps_data_file = os.path.join(self.imu_gps_output_dir, 'gps_data.txt')

        # Verificar si el archivo existe
        if os.path.exists(tf_data_file):
            # Eliminar el archivo si existe
            os.remove(tf_data_file)
            print("Archivo tf_data.txt reiniciado correctamente.")
        else:
            print("El archivo tf_data.txt no existe.")

        # Verificar si el archivo existe
        if os.path.exists(gps_data_file):
            # Eliminar el archivo si existe
            os.remove(gps_data_file)
            print("Archivo gps_data.txt reiniciado correctamente.")
        else:
            print("El archivo gps_data.txt no existe.")

        self.cams_params = {}  # for all usb cameras

        # Check if rosbag is compressed
        if rosbag_path.endswith(".zstd"):
            decompressed_rosbag_path = rosbag_path.replace(".zstd", "")
            decompress_file(rosbag_path, decompressed_rosbag_path)
            self.rosbag_path = decompressed_rosbag_path

        storage_options, converter_options = get_rosbag_options(self.rosbag_path)

        self.rosbag = rosbag2_py.SequentialReader()
        self.rosbag.open(storage_options, converter_options)
        topic_types = self.rosbag.get_all_topics_and_types()
        # Create a map for quicker lookup
        self.topic_types_map = {
            topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
        }
        for topic, msg_type in self.topic_types_map.items():
            print(f"Topic: {topic}, Type: {msg_type}")
        # print(self.topic_types_map)

        if not set(self.topics).issubset(set(self.topic_types_map.keys())):
            raise ValueError(
                "The topics you provided are not in the bag file. "
                "Please check the topics you provided and the bag file. "
                f"Topics in bag: {list(self.topic_types_map.keys())}"
            )

        # Check if /tf is in the topics list
        if '/tf' in self.topics:
            # Remove '/tf' from the topics list
            self.topics.remove('/tf')
            # Add /tf/odom and /tf/base_link if /tf is present
            self.topics.extend(['/tf/odom', '/tf/base_link'])

        # Idea taken from C++ example http://wiki.ros.org/rosbag/Cookbook#Analyzing_Stereo_Camera_Data
        self.filters_dict = {topic: message_filters.SimpleFilter() for topic in self.topics} #se modifica topics por self.topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            list(self.filters_dict.values()), queue_size, time_delta
        )

        self.ts.registerCallback(self.sync_callback)

        # tqdm progress bar
        if self.verbose:
            self.pbar = tqdm(desc="Serializing rosbag data... ")

        self.video_writers = {}
        self.i = 0  # Inicializar el contador


    def sync_callback(self, *msgs):
        """! Callback for the approximate time synchronizer of msgs
        @param msgs (list of msgs) list of msgs from the topics
        """
        print("callback")
        if self.verbose:
            self.pbar.update(1)

        img_data = {}
        imu_gps_data = {}
        gps_data = {}
        tf_data = {}

        

        # Iterate over arguments, each argument is a different msg
        for topic, msg in zip(self.topics, msgs):
            print("sale zip")
            print("Topics:", self.topics)
            print("Message types:", [type(msg) for msg in msgs])
            # Parse msg depending on its type
            msg_info_dict = self.parse_msg(msg, topic)
            
            # Get the timestamp of the message
            # if topic == "/camera/color/image_raw":
            # if topic.endswith("/image_raw"):
            if "/image_raw" in topic:
                timestamp = msg.header.stamp if hasattr(msg.header, 'stamp') else None

            # Print the topic, timestamp, and any other relevant information
            #print(f"Topic: {topic}, Timestamp: {timestamp}")
            # if self.is_gps == True:
            #     if isinstance(msg, sensor_msgs.msg.NavSatFix):
            #         gps_data = msg_info_dict.pop("data")
            #         if gps_data:
            #             imu_gps_data['gps'] = gps_data
                
            #     if gps_data:
            #             gps_file_path = os.path.join(self.imu_gps_output_dir, "gps_data.txt")
            #             image_filename = f"{folder_name}_image_{self.i+1:04d}.jpg"
            #             with open(gps_file_path, "a") as gps_file:
            #                 gps_info = f"{image_filename} {gps_data['latitude']} {gps_data['longitude']} {gps_data['altitude']}\n"
            #                 gps_file.write(gps_info)
            #                 #gps_file.write(f"{image_filename}, GPS Data: {gps_data}\n")
            #     # Save updated EXIF data back to the image
            #             exif_bytes = piexif.dump(exif_dict)
            #             piexif.insert(exif_bytes, image_path)
            #             imu_gps_data.pop("gps")

            if isinstance(msg, sensor_msgs.msg.CompressedImage) or isinstance(msg, sensor_msgs.msg.Image):
                #img_data[topic] = msg_info_dict.get("data")
                print("asigna img_data")
                img_data[topic] = msg_info_dict.pop("data")
                print(topic)
                print(type(msg))
                print(img_data[topic])

                # Mostrar imagenes
                #cv2.imshow(topic, img_data[topic])
                #cv2.waitKey(1)

                # # Guardar las imágenes descomprimidas en el directorio de salida
                # for topic, img_data in img_data.items():
                #     #image_filename = f"{topic}_{msg.header.stamp}.jpg"
                #     #image_filename = f"{msg.header.stamp}.jpg"
                #     image_filename = f"image_{self.i+1:04d}.jpg"
                #     image_path = os.path.join(self.output_dir, image_filename)
                #     cv2.imwrite(image_path, img_data) 
            # elif isinstance(msg, sensor_msgs.msg.Imu):
            #     imu_data = msg_info_dict.get("data")
            #     if imu_data:
            #         imu_file_path = os.path.join(self.imu_gps_output_dir, "imu_data.txt")
            #         with open(imu_file_path, "a") as imu_file:
            #             # imu_file.write(f"Timestamp: {timestamp}, IMU Data: {imu_data}\n")
            #             imu_file.write(f"{image_filename}, IMU Data: {imu_data}\n")
            elif isinstance(msg, sensor_msgs.msg.NavSatFix):
                gps_data = msg_info_dict.pop("data")
                if gps_data:
                    imu_gps_data['gps'] = gps_data
                # if gps_data:
                #     gps_file_path = os.path.join(self.imu_gps_output_dir, "gps_data.txt")
                #     with open(gps_file_path, "a") as gps_file:
                #         gps_file.write(f"{image_filename}, GPS Data: {gps_data}\n")
            elif isinstance(msg, geometry_msgs.msg.TransformStamped):
                #print("process tf")
                tf_data = msg_info_dict.pop("data")
                if "/tf/odom" in topic:
                    print("process tf odom")
                    x_odom = tf_data["translation"]["x"]
                    y_odom = tf_data["translation"]["y"]
                    z_odom = tf_data["translation"]["z"]
                    rx_odom = tf_data["rotation"]["x"]
                    ry_odom = tf_data["rotation"]["y"]
                    rz_odom = tf_data["rotation"]["z"]
                    w_odom = tf_data["rotation"]["w"]
                    # Convertir el cuaternión a ángulos de Euler (en radianes)
                    q = np.array([w_odom, rx_odom, ry_odom, rz_odom])
                    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
                    cosy_cosp = 1 - 2 * (q[2] ** 2 + q[3] ** 2)
                    rz_e_odom = np.arctan2(siny_cosp, cosy_cosp)
                    #euler_angles_odom = tf.euler_from_quaternion([rx_odom, ry_odom, rz_odom, w_odom])
                    print(rz_e_odom)


                if "/tf/base_link" in topic:
                    print("process tf base_link")
                    x_base_link = tf_data["translation"]["x"]
                    y_base_link = tf_data["translation"]["y"]
                    z_base_link = tf_data["translation"]["z"]
                    rx_base_link = tf_data["rotation"]["x"]
                    ry_base_link = tf_data["rotation"]["y"]
                    rz_base_link = tf_data["rotation"]["z"]
                    w_base_link = tf_data["rotation"]["w"]
                    # Convertir el cuaternión a ángulos de Euler (en radianes)
                    q = np.array([w_base_link, rx_base_link, ry_base_link, rz_base_link])
                    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
                    cosy_cosp = 1 - 2 * (q[2] ** 2 + q[3] ** 2)
                    rz_e_base_link = np.arctan2(siny_cosp, cosy_cosp)
                    #euler_angles_odom = tf.euler_from_quaternion([rx_odom, ry_odom, rz_odom, w_odom])
                    print(rz_e_base_link)
                    # euler_angles_base_link = tf.euler_from_quaternion([rx_base_link, ry_base_link, rz_base_link, w_base_link])
                    # print(euler_angles_base_link)


                
                print(tf_data["translation"]["x"])
                print(tf_data["translation"]["y"])

        
        #plt.ion()

        # Guardar las imágenes en el directorio de salida con intrinsics y GPS
        for topic, img_data in img_data.items():
            print("img data items")
            print(topic)
            if "/video_mapping/left/image_raw" in topic:
                folder_name = "left"
            elif "/video_mapping/right/image_raw" in topic:
                folder_name = "right"
            elif "/camera/color/image_raw" in topic:
                folder_name = "front"
            else:
                folder_name = "unknown"
            
            if not os.path.exists(os.path.join(self.output_dir, folder_name)):
                os.makedirs(os.path.join(self.output_dir, folder_name))

            image_filename = f"{folder_name}_image_{self.i+1:04d}.jpg"
            image_path = os.path.join(self.output_dir, folder_name, image_filename)
            if not self.is_gps:
                cv2.imwrite(image_path, img_data)
                exif_dict = piexif.load(image_path)
            # Agregar información sobre la distancia focal y los puntos centrales
            fx = self.cams_params[topic]["k"][0, 0]
            fy = self.cams_params[topic]["k"][1, 1]
            cx = self.cams_params[topic]["k"][0, 2]
            cy = self.cams_params[topic]["k"][1, 2]
            if not self.is_gps:
                exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (int(fx*1000), 1000)  # Distancia focal en milímetros
            #exif_dict["Exif"][piexif.ExifIFD.PixelXDimension] = int(cx)  # Punto central en el eje x
            #exif_dict["Exif"][piexif.ExifIFD.PixelYDimension] = int(cy)  # Punto central en el eje y


            # Guardar la información de GPS en el EXIF de la imagen
            if 'gps' in imu_gps_data:
                #exif_dict = piexif.load(image_path)
                latitude = imu_gps_data['gps']['latitude']
                longitude = imu_gps_data['gps']['longitude']
                altitude = imu_gps_data['gps']['altitude']
                lat = abs(latitude)
                lon = abs(longitude)
                lat_deg = int(lat)
                lat_min = int((lat - lat_deg) * 60)
                lat_sec = int(((lat - lat_deg) * 60 - lat_min) * 60 * 1000)
                lon_deg = int(lon)
                lon_min = int((lon - lon_deg) * 60)
                lon_sec = int(((lon - lon_deg) * 60 - lon_min) * 60 * 1000)
                # Asignar valores de latitud, longitud y altitud al diccionario EXIF
                if not self.is_gps:
                    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N' if latitude >= 0 else 'S'
                    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = ((lat_deg, 1), (lat_min, 1), (lat_sec, 1000))
                    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'E' if longitude >= 0 else 'W'
                    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = ((lon_deg, 1), (lon_min, 1), (lon_sec, 1000))
                    exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (int(altitude*1000), 1000)  # Altitud en metros
                    print(exif_dict)

            #print(exif_dict)
            if gps_data:
                    gps_file_path = os.path.join(self.imu_gps_output_dir, "gps_data.txt")
                    with open(gps_file_path, "a") as gps_file:
                        gps_info = f"{image_filename} {gps_data['latitude']} {gps_data['longitude']} {gps_data['altitude']}\n"
                        gps_file.write(gps_info)
                        #gps_file.write(f"{image_filename}, GPS Data: {gps_data}\n")
            # Save updated EXIF data back to the image
                    if not self.is_gps:
                        exif_bytes = piexif.dump(exif_dict)
                        piexif.insert(exif_bytes, image_path)
                        imu_gps_data.pop("gps")

            if tf_data:
                #transformada desde map(0,0) a base_link: map-odom-base_link
                ###
                # # Convertir el cuaternión a una matriz de rotación
                # print("matrices de rotacion")
                # #R_odom = quat.quat2mat([w_odom, rx_odom, ry_odom, rz_odom])
                # R_odom = quat.quat2mat([1, 0, 0, 0])
                
                # R_base_link = quat.quat2mat([w_odom, rx_odom, ry_odom, rz_odom])
                
                # #R_base_link = quat.quat2mat([w_base_link, rx_base_link, ry_base_link, rz_base_link])
                # print(R_odom)
                # print(R_base_link)
                # # Armar matrices de transformación con rotación y traslación
                # # Construir la matriz de transformación map-odom
                # T_odom = np.eye(4)
                # T_odom[:3, :3] = R_odom
                # T_odom[0, 3] = x_odom
                # T_odom[1, 3] = y_odom
                # T_odom[2, 3] = z_odom
                # # Construir la matriz de transformación odom-base_link
                # T_base_link = np.eye(4)
                # T_base_link[:3, :3] = R_base_link
                # T_base_link[0, 3] = x_base_link
                # T_base_link[1, 3] = y_base_link
                # T_base_link[2, 3] = z_base_link
                # print("matrices de transformacion")
                # print(T_odom)
                # print(T_base_link)
                # # Multiplicar las matrices
                # pose_odom = np.dot(T_odom, np.array([0, 0, 0, 1]))
                # pose = np.dot(T_base_link, pose_odom)
                # print("vector de posicion")
                # print(pose)
                ###

                # pose_x = x_odom + (x_base_link * np.cos(rz_e_odom))
                # pose_y = y_odom + x_base_link * np.sin(rz_e_odom)
                pose_odom = np.array([x_odom, y_odom])
                Rz_odom = np.array([[np.cos(rz_e_odom), -np.sin(rz_e_odom)],
                                    [np.sin(rz_e_odom), np.cos(rz_e_odom)]])
                # Traslación en base_link
                pose_base_link = np.array([x_base_link, y_base_link])
                R_base_link = np.dot(Rz_odom, pose_base_link)

                pose_x = pose_odom[0] + R_base_link[0]
                pose_y = pose_odom[1] + R_base_link[1]
                #self.pose_x_values.append(pose_x)
                #self.pose_y_values.append(pose_y)

                

                # pose_x = pose[0]
                # pose_y = pose[1]
                # pose_x = x_base_link
                # pose_y = y_base_link
                pose_z = 0
                print(pose_x)
                print(pose_y)
                print(pose_z)

                #enmascarar coordenadas cartesianas como GPS para poder guardar en exif
                deg_x = int(pose_x)
                min_x = int((pose_x - deg_x) * 60 * 10000)
                deg_y = int(pose_y)
                min_y = int((pose_y - deg_y) * 60 * 10000)
                # GPS de EXIF solo acepta numeros positivos por lo que el bag tiene que grabarse sobre el lado positivo del mapa
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = ((deg_x, 1), (min_x, 10000), (0, 1))
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N' if pose_y >= 0 else 'S'
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = ((deg_y, 1), (min_y, 10000), (0, 1))
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'E' if pose_x >= 0 else 'W'
                exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (int(pose_z*1000), 1000)  # Altitud en metros
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = str(timestamp).encode("utf-8")
                #timestamp
                print(exif_dict)
                # se define un valor cualquiera de altitud para que el GPS pueda ser interpretado por COLMAP
                # COLMAP guarda estos datos con una prcisión de mm
                # COLMAP muestra la informacion en grados decimales,
                # aunque realmente provienen de coordenadas cartesinas con "map" como referencia
                # Ejm: LAT=78.753 (metros desde el 0,0 del map del rosbag), LON=103.704, ALT=1500.000
                tf_file_path = os.path.join(self.imu_gps_output_dir, "tf_data.txt")
                with open(tf_file_path, "a") as tf_file:
                    tf_info = f"{image_filename} {pose_x} {pose_y} {pose_z}\n"
                    tf_file.write(tf_info)
                    #gps_file.write(f"{image_filename}, GPS Data: {gps_data}\n")
                # Save updated EXIF data back to the image
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, image_path)
                print(exif_dict)
                
        # self.i += 1
        # Mostrar la imagen

        # plt.show()
        # plt.ioff()

    def parse_msg(self, msg: tp.Any, topic: str) -> tp.Dict[str, tp.Any]:
        """
        Parses msg depending on its type.
        @param msg (any) message to be parsed
        @param topic (str) topic of the message
        @return (dict) message as dictionary
        """
        if isinstance(msg, sensor_msgs.msg.Image):
            return self.parse_image(msg, topic)
        elif isinstance(msg, sensor_msgs.msg.CompressedImage):
            return self.parse_compressed_image(msg, topic)
        elif isinstance(msg, sensor_msgs.msg.CameraInfo):
            return self.parse_camera_info(msg)
        elif isinstance(msg, sensor_msgs.msg.Imu):
            return self.parse_imu_info(msg)
        elif isinstance(msg, sensor_msgs.msg.NavSatFix):
            return self.parse_gps_info(msg)
        elif isinstance(msg, geometry_msgs.msg.TransformStamped):
           return self.parse_tf_info(msg)
        else:
            raise ValueError(f"Unsupported message type {type(msg)}")
        
    
    def parse_camera_info(
        self, msg: sensor_msgs.msg.CameraInfo
    ) -> tp.Dict[str, tp.Any]:
        """Parses camera info msg and saves it in self.stereo_cam_model
        @param msg (CameraInfo) camera info message
        @param topic (str) topic of the message
        @return (dict) empty dict
        """
        params = {}

        params["dim"] = (int(msg.width), int(msg.height))
        params["k"] = np.array(msg.k).reshape((3, 3))
        params["p"] = np.array(msg.p).reshape((3, 4))
        params["distortion_model"] = msg.distortion_model
        params["d"] = np.array(msg.d)
        params["r"] = np.array(msg.r).reshape((3, 3))

        if params["distortion_model"] in ("equidistant", "fisheye"):
            initUndistortRectifyMap_fun = cv2.fisheye.initUndistortRectifyMap
        else:
            initUndistortRectifyMap_fun = cv2.initUndistortRectifyMap
        params["map1"], params["map2"] = initUndistortRectifyMap_fun(
            params["k"],
            params["d"],
            np.eye(3),
            params["k"],
            params["dim"],
            cv2.CV_16SC2,
        )

        return params
    
    def scale_calibration(self, params: dict, factor: int) -> dict:
        """! Scale calibration parameters

        @param params (dict) calibration parameters
        @param factor (int) scale factor

        @return scaled calibration parameters
        """
        if factor != 1:
            # params["k"] = params["k"] * factor
            # params["k"][2, 2] = 1
            # params["dim"] = params["dim"] * factor

            params["k"][0, 0] *= factor  # Escalando fx
            params["k"][1, 1] *= factor  # Escalando fy
            params["k"][0, 2] *= factor  # Escalando cx
            params["k"][1, 2] *= factor  # Escalando cy
            params["dim"] = (params["dim"][0] * factor, params["dim"][1] * factor)  # Escalando dimensiones de la imagen
        if factor == 1:
            # params["k"] = params["k"] * factor
            # params["k"][2, 2] = 1
            # params["dim"] = params["dim"] * factor

            params["k"][0, 0] = 382.9  # Escalando fx
            params["k"][1, 1] = 382.9  # Escalando fy
            params["k"][0, 2] = 320  # Escalando cx
            params["k"][1, 2] = 240  # Escalando cy
            params["dim"] = (640, 480)  # Escalando dimensiones de la imagen
            # params["k"][0, 2] = 640  # Escalando cx
            # params["k"][1, 2] = 360  # Escalando cy
            # params["dim"] = (1280, 720)  # Escalando dimensiones de la imagen

        if params["distortion_model"] in ("equidistant", "fisheye"):
            initUndistortRectifyMap_fun = cv2.fisheye.initUndistortRectifyMap
        else:
            initUndistortRectifyMap_fun = cv2.initUndistortRectifyMap
        params["map1"], params["map2"] = initUndistortRectifyMap_fun(
            params["k"],
            params["d"],
            params["r"],
            params["k"],
            params["dim"],
            # params["dim"] * factor,
            cv2.CV_16SC2,
        )
        return params

    def parse_image(
        self, msg: sensor_msgs.msg.Image, topic: str
    ) -> tp.Dict[str, tp.Any]:
        """
        Parses image message
        @param msg (Image) image message
        @param topic (str) topic of the message
        @return (dict) image message as dictionary with the path to the image
            and the image itself as numpy array
        """
        # Convert image to cv2 image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # only undistort usb camera images, stereo is already undistorted
        if self.cams_params.get(topic) and self.undistort:
            print("distorted")
            cv_image = cv2.remap(
                cv_image,
                self.cams_params[topic]["map1"],
                self.cams_params[topic]["map2"],
                cv2.INTER_LINEAR,
            )
        return {"data": cv_image}
    
    def parse_compressed_image(
        self, msg: sensor_msgs.msg.CompressedImage, topic: str
    ) -> tp.Dict[str, tp.Any]:
        np_arr = np.array(msg.data, dtype=np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # only undistort usb camera images, stereo is already undistorted
        if self.cams_params.get(topic) and self.undistort:
            cv_image = cv2.remap(
                cv_image,
                self.cams_params[topic]["map1"],
                self.cams_params[topic]["map2"],
                cv2.INTER_LINEAR,
            )
        return {"data": cv_image}

    def parse_imu_info(self, msg: sensor_msgs.msg.Imu) -> tp.Dict[str, tp.Any]:
        """Parses IMU message."""
        imu_data = {
            "orientation": {
                "x": msg.orientation.x,
                "y": msg.orientation.y,
                "z": msg.orientation.z,
                "w": msg.orientation.w,
            },
            "angular_velocity": {
                "x": msg.angular_velocity.x,
                "y": msg.angular_velocity.y,
                "z": msg.angular_velocity.z,
            },
            "linear_acceleration": {
                "x": msg.linear_acceleration.x,
                "y": msg.linear_acceleration.y,
                "z": msg.linear_acceleration.z,
            }
        }
        return {"data": imu_data}
    
    def parse_gps_info(self, msg: sensor_msgs.msg.NavSatFix) -> tp.Dict[str, tp.Any]:
        """Parses GPS message."""
        gps_data = {
            "latitude": msg.latitude,
            "longitude": msg.longitude,
            "altitude": msg.altitude
        }
        return {"data": gps_data}
    
    def parse_tf_info(self, msg: geometry_msgs.msg.TransformStamped) -> tp.Dict[str, tp.Any]:
        """Parses TF message."""
        tf_data = {
            "child": msg.child_frame_id,
            "translation": {
                "x": msg.transform.translation.x,
                "y": msg.transform.translation.y,
                "z": msg.transform.translation.z,
            },
            "rotation": {
                "x": msg.transform.rotation.x,
                "y": msg.transform.rotation.y,
                "z": msg.transform.rotation.z,
                "w": msg.transform.rotation.w,
            }
        }
        return {"data": tf_data}

    
    def process_rosbag(self):
        """Processes the rosbag file, it starts reading message by
        message and sending them to the approximate synchronizer that
        will trigger the synchronization of the messages function.
        """

        camera_info_topics = [topic for topic in self.topics if "image_raw" in topic]
        camera_info_topics = [topic.replace("image_raw", "camera_info") for topic in camera_info_topics]
        self.is_gps = False

        # Diccionario para rastrear si se ha encontrado al menos un mensaje de cada topic de camera_info_topics
        found_data_for_topics = {topic: False for topic in camera_info_topics}

        # Read rosbag until all camera info needed are found
        while self.rosbag.has_next() and not all(found_data_for_topics.values()):
            (topic, data, t) = self.rosbag.read_next()
            print(topic)
            
            if topic not in camera_info_topics:
                continue

            msg_type = get_message(self.topic_types_map[topic])
            msg = deserialize_message(data, msg_type)
            #print(topic)

            # if topic == "/camera/color/image_raw":
            #     self.i += 1
            corresponding_topic = topic.replace("camera_info", "image_raw")
            self.cams_params[corresponding_topic] = self.parse_camera_info(msg)
            if corresponding_topic == "/video_mapping/right/image_raw":
                self.cams_params[corresponding_topic] = self.scale_calibration(self.cams_params[corresponding_topic], 2)
            if corresponding_topic == "/video_mapping/left/image_raw":
                self.cams_params[corresponding_topic] = self.scale_calibration(self.cams_params[corresponding_topic], 1)
            

            # Marcar el topic como encontrado
            found_data_for_topics[topic] = True

        print(self.cams_params)
        
        # Continue reading the rosbag
        # We will lost some frames but this way all will have instrinsincs
        # Another way to do this keeping all frames is by creating a new SequentialReader object
        # Try with seek()
        self.rosbag.seek(0)
        print("test before read images")
        print("topics")
        print(self.topics)
        print("filter dict")
        print(self.filters_dict)

        # acá crear un ciclo for que reinicie el seek para cada topic de camara requerido y que modifique el diccionario
        # de topics filtrados. Adentro del for va el while
        for image_raw_topic in [topic for topic in self.topics if '/image_raw' in topic]:
            # Actualizar self.topics para sincronizar solo un /image_raw con /odom y /base_link
            self.topics = [image_raw_topic, '/tf/odom', '/tf/base_link']
            
            # Actualizar self.filters_dict para incluir los nuevos temas
            #self.filters_dict.update({topic: message_filters.SimpleFilter() for topic in self.topics})
            self.filters_dict = {topic: message_filters.SimpleFilter() for topic in self.topics}
            # Eliminar los filtros obsoletos de self.filters_dict
            for topic in list(self.filters_dict.keys()):
                if topic not in self.topics:
                    del self.filters_dict[topic]
            self.ts = message_filters.ApproximateTimeSynchronizer(list(self.filters_dict.values()), self.queue_size, self.time_delta)
            self.ts.registerCallback(self.sync_callback)
            self.rosbag.seek(0)  # Reiniciar la lectura al principio
            self.i = 0

            print("test in for of read images")
            print("topics")
            print(self.topics)
            print("filter dict")
            print(self.filters_dict)


            while self.rosbag.has_next():
                (topic, data, t) = self.rosbag.read_next()
                
                skip_iteration = False

                #this is to avoid reading customized messages that can show errors
                if topic not in self.topics:
                    if topic != "/tf":
                        continue

                print(topic)
                msg_type = get_message(self.topic_types_map[topic])
                msg = deserialize_message(data, msg_type)
                
                # crear condicional para cambiar el nombre del topic /tf según el child_frame_id a /tf/odom y
                # /tf/base_linkpara sincronizar los mensajes de odom y base_link de tf como topics separados
                #print("inicio test")
                #print(msg_type)
                if isinstance(msg, tf2_msgs.msg.TFMessage):
                    for transform in msg.transforms:
                        #print(transform.child_frame_id)
                        if transform.child_frame_id in ('odom'):
                            #print(topic)
                            topic = topic.replace("tf", "tf/odom")
                            print(topic)
                            #print(msg)
                            msg = transform
                            # se convierte de tf2_msgs.msg.TFMessage a geometry_msgs.msg.TransformStamped
                            #print("espacio")
                            #print(msg)
                        elif transform.child_frame_id in ('base_link'):
                            #print(topic)
                            topic = topic.replace("tf", "tf/base_link")
                            print(topic)
                            msg = transform
                            #print("in tf test")
                        else:
                            skip_iteration = True
                if skip_iteration:
                    continue

                #print("fin test")
                

                # si no es odom o base_link pasa el while

                if topic in self.filters_dict:
                    print("in filters_dict input")
                    print(topic)
                    print("in filters_dict output")
                    filter_obj = self.filters_dict[topic]
                    filter_obj.signalMessage(msg)
                    #if topic == "/camera/color/image_raw":
                    if topic.endswith("/image_raw"):
                        self.i += 1


        # # Guardar ID de impagenes sincronizado con GPS en archivo de texto
        # self.is_gps = True
        # for image_raw_topic in [topic for topic in self.original_topics if '/image_raw' in topic]:
        #     # Actualizar self.topics para sincronizar solo un /image_raw con /odom y /base_link
        #     self.topics = [image_raw_topic, '/fix']
            
        #     # Actualizar self.filters_dict para incluir los nuevos temas
        #     #self.filters_dict.update({topic: message_filters.SimpleFilter() for topic in self.topics})
        #     self.filters_dict = {topic: message_filters.SimpleFilter() for topic in self.topics}
        #     # Eliminar los filtros obsoletos de self.filters_dict
        #     for topic in list(self.filters_dict.keys()):
        #         if topic not in self.topics:
        #             del self.filters_dict[topic]
        #     self.ts = message_filters.ApproximateTimeSynchronizer(list(self.filters_dict.values()), self.queue_size, self.time_delta)
        #     self.ts.registerCallback(self.sync_callback)
        #     self.rosbag.seek(0)  # Reiniciar la lectura al principio
        #     self.i = 0

        #     print("Sync images with GPS")

        #     while self.rosbag.has_next():
        #         (topic, data, t) = self.rosbag.read_next()
                
        #         skip_iteration = False

        #         #this is to avoid reading customized messages that can show errors
        #         if topic not in self.topics:
        #             if topic != "/tf":
        #                 continue

        #         print(topic)
        #         msg_type = get_message(self.topic_types_map[topic])
        #         msg = deserialize_message(data, msg_type)

        #         if topic in self.filters_dict:
        #             print("in filters_dict input")
        #             print(topic)
        #             print("in filters_dict output")
        #             filter_obj = self.filters_dict[topic]
        #             filter_obj.signalMessage(msg)
        #             #if topic == "/camera/color/image_raw":
        #             if topic.endswith("/image_raw"):
        #                 self.i += 1
        # self.is_gps = False


            # DEBUG
            # if isinstance(msg, sensor_msgs.msg.Image):
            #     cv_img = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
            #     cv2.imshow(f"{topic}-debug", cv_img)
            #     cv2.waitKey(1)
                
        # # Crear una nueva figura y ejes
        # fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # # Graficar solo x
        # axes[0].plot(self.pose_x_values)
        # axes[0].set_title('Pose X')

        # # Graficar solo y
        # axes[1].plot(self.pose_y_values)
        # axes[1].set_title('Pose Y')

        # # Graficar x vs y
        # axes[2].plot(self.pose_x_values, self.pose_y_values, 'bo')  # 'bo' para puntos azules
        # axes[2].set_title('Pose X vs Pose Y')
        # axes[2].set_xlabel('Pose X')
        # axes[2].set_ylabel('Pose Y')

        # # Ajustar el espaciado entre subgráficas
        # plt.tight_layout()

        # # # Mostrar la imagen
        # plt.show()

def create_masks():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8x-seg.pt')

    # Directorio base de entrada y salida
    input_base_folder = '/working/colmap_ws/images'
    output_base_folder = '/working/colmap_ws/masks'

    # Iterar sobre las carpetas dentro del directorio base de entrada
    for folder_name in os.listdir(input_base_folder):
        # Obtener la ruta completa de la carpeta de entrada y salida para esta iteración
        input_folder = os.path.join(input_base_folder, folder_name)
        output_folder = os.path.join(output_base_folder, folder_name)

        # Crear la carpeta de salida si no existe
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Obtener la lista de archivos en el directorio de entrada y ordenarlos alfabéticamente
        files = os.listdir(input_folder)
        files.sort()

        # Iterar sobre todas las imágenes en el directorio de entrada
        for filename in files:
            if filename.endswith('.jpg'):
                # Cargar la imagen original
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                # Ejecutar la segmentación en la imagen original
                results = model.predict(image, save=False, imgsz=[736,1280])

                # Inicializar la máscara combinada
                combined_mask = torch.zeros([736,1280], dtype=torch.uint8).to('cuda')

                # Iterar sobre los resultados
                for result in results:
                    # Verificar si se detectaron máscaras en el resultado
                    if result.masks is not None:
                        # Extraer las máscaras y las cajas de detección
                        masks = result.masks.data.to('cuda')
                        boxes = result.boxes.data.to('cuda')

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

def create_image_lists(overlap=10):
    input_folder= "/working/colmap_ws/images"
    output_base_folder= "/working/colmap_ws/lists_folder"
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

def main(
    sync_topics: tp.List[str] = [
        # include the image_raw and the camera_info of the cameras you want to synchronize
        # include /fix if you want GPS data
        "/camera/color/image_raw",
        #"/camera/color/camera_info",
        #"/video_mapping/left/image_raw",
        #"/video_mapping/left/camera_info",
        "/video_mapping/right/image_raw",
        #"/video_mapping/right/camera_info",
        #"/video_mapping/rear/image_raw",
        #"/video_mapping/rear/camera_info",
        #"/imu/data",
        #"/fix",
        "/tf"
    ],
    fps: int = 10,
    undistort: bool = True,
    debug: bool = False,
    imshow: bool = False,
):
    """
    Main function for processing the rosbag files.
    @sync_topics: list of topics to process
    @bag_path: list of paths to the rosbag files to process
    @fps: fps of the output videos
    @undistort: if True, undistorts the images
    @concat_images: if True, concatenates the images horizontally
    @debug: if True, waits for debugger to attach
    @imshow: if True, shows the images
    """

    bag_path = os.path.abspath(os.path.expanduser("/working/main_folder/sfm_0.mcap"))
    
    if debug:
        import debugpy  # pylint: disable=import-error

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    rosbag_serializer = RosBagSerializer(
        bag_path,
        sync_topics,
        time_delta=0.1,
        fps=fps,
        undistort=undistort,
        imshow=imshow,
    )
    rosbag_serializer.process_rosbag()
    create_masks()
    create_image_lists()


if __name__ == "__main__":
    typer.run(main)
