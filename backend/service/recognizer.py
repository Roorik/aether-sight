# медиатор
from io import BytesIO
from backend.service.processor import Processor

class Recognizer():
    '''
    Управляющий класс для работы с видео и детекцией объектов.
    '''
    def execute(self, stream: BytesIO):
        Processor().open_video(stream)
        