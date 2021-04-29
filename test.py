import os
import asyncio
import cv2
import paddlehub as hub

from typing import List, Optional, Union

from wechaty_puppet import FileBox  # type: ignore
from wechaty_puppet import MessageType

from wechaty import Wechaty, Contact
from wechaty.user import Message, Room

os.environ['WECHATY_PUPPET_SERVICE_ENDPOINT'] = '127.0.0.1:8080'
os.environ['WECHATY_PUPPET_SERVICE_TOKEN'] = 'python-wechaty-d1027e0a-cd4d-458e-83d5-a381c70b0726'

# 加载移动端预训练模型
ocr = hub.Module(name="chinese_ocr_db_crnn_server")



class MyBot(Wechaty):

    async def on_message(self, msg: Message):
        """
        listen for message event
        """
        from_contact: Optional[Contact] = msg.talker()
        text = msg.text()
        room: Optional[Room] = msg.room()


        if msg.is_self():
            return


        if text == 'ocr':
            conversation: Union[
                Room, Contact] = from_contact if room is None else room
            await conversation.ready()
            await conversation.say('please send an image')



        if msg.type() == MessageType.MESSAGE_TYPE_IMAGE:
            conversation: Union[
                Room, Contact] = from_contact if room is None else room

            img = await msg.to_file_box()
            # await conversation.say(f'./{img.name}') # for test use
            await img.to_file(f'./received_files/{img.name}')

            np_images = [cv2.imread(f'./received_files/{img.name}')]

            results = ocr.recognize_text(
            images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
            use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
            output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
            visualization=True,       # 是否将识别结果保存为图片文件；
            box_thresh=0.5,           # 检测文本框置信度的阈值；
            text_thresh=0.5)          # 识别中文文本置信度的阈值；

            # in case of error of paddleocr
            if results is None:
                return

            data = results[0]['data']
            save_path = results[0]['save_path']
            s = ""
            for information in data:
                s += information['text']
                s += '\n'

            file_box = FileBox.from_file(save_path)
            await conversation.say(file_box)
            await conversation.say(s)


#asyncio.run(MyBot().start())
async def main():
    bot = MyBot()
    await bot.start()

asyncio.run(main())
