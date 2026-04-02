# utils/timer.py

import time


class Timer:

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):

        if self.start_time is None:
            return "00:00:00"

        elapsed = int(time.time() - self.start_time)

        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"