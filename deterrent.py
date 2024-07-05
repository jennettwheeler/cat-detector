import os.path
import time
import hashlib
import pygame
import random

from gtts import gTTS


class Deterrent:
    def __init__(self, min_interval, messages):
        self.last_message_file = None
        self.message_files = []
        for message in messages:
            filename = f"{hashlib.md5(message.encode('utf-8')).hexdigest()}.mp3"
            if not os.path.exists(filename):
                print(f"Creating {filename} for message: {message}")
                message_mp3 = gTTS(text=message, lang='en', slow=False)
                message_mp3.save(filename)
            self.message_files.append(filename)

        self.min_interval = min_interval
        self.last_time = time.time() - min_interval

    def try_deter(self, position):
        current_time = time.time()
        if current_time - self.last_time >= self.min_interval:
            self.last_time = current_time
            self.deter(position)


    def deter(self, position):
        message_file = random.choice(self.message_files)
        while len(self.message_files) > 1 and message_file == self.last_message_file:
            message_file = random.choice(self.message_files)
        self.last_message_file = message_file

        pygame.mixer.init()
        pygame.mixer.music.load(message_file)
        pygame.mixer.music.play()