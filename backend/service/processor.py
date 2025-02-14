import cv2
import imageio
import numpy as np

from io import BytesIO
from pathlib import Path
from cv2.typing import Size

from ultralytics import YOLO

BASE_URL = Path(__file__).parent
res_dir = BASE_URL.joinpath('res')
video_dir = BASE_URL.parent.parent.joinpath('test2.mp4')

class Processor():
    """Класс для обработки видео и обнаружения объектов с помощью YOLO."""
    
    def __init__(self, model_path: str = 'yolo11n.pt'):
        """Инициализация детектора с моделью YOLO."""
        self.model = YOLO(model_path)
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Преобразует кадр в формат, подходящий для модели.
        
        :param frame: Входное изображение (BGR).
        :return: Нормализованный кадр (RGB) с добавленным батч-измерением.
        """
        resized_frame = cv2.resize(frame, (224, 224))  # Размеры могут варьироваться
        normalized_frame = resized_frame / 255.0  # Нормализация пикселей
        return np.expand_dims(normalized_frame, axis=0)  # Добавление батч-измерения

    def postprocess_result(self, frame, processed_data):
        """
        Обрабатывает результаты модели и визуализирует их на кадре.
        """
        # Пример: отрисовка результатов на кадре
        if processed_data.get("detections"):
            for detection in processed_data["detections"]:
                x, y, w, h = detection["bbox"]  # Пример структуры данных
                confidence = detection["confidence"]
                label = detection["label"]

                # Отрисовка рамок и текста на изображении
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def open_video(self, content: BytesIO):
    # Читаем бинарные данные из BytesIO
        content.seek(0)
        reader = imageio.get_reader(content, format="mp4")  # Читаем видео из памяти
    
        # Получаем параметры видео
        meta_data = reader.get_meta_data()
        fps = meta_data["fps"]
        width, height = meta_data["size"]

     
        # Получение параметров исходного видео
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Кодек для сохранения видео

        model = YOLO('yolo11n.pt')
        
        detected_objects = []
            
        for frame in reader:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Преобразуем в BGR для OpenCV
            results = model.predict(frame)

            for result in results:
                for box in result.boxes:
                    c = box.cls
                    detected_objects.append(model.names[int(c)])

        
        print(detected_objects)

        reader.close()
        return detected_objects
    