import threading


class Buffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        lock = threading.RLock()
        self.has_data = threading.Condition(lock)
        self.has_pos = threading.Condition(lock)

    def get_size(self):
        return self.size

    def get(self):
        """
        Get data from this buffer. If the buffer is empty then the current thread is blocked until there is
        at least one data available.
        :return:
        """
        with self.has_data:
            while len(self.buffer) == 0:
                self.has_data.wait()
            result = self.buffer[0]
            del self.buffer[0]
            self.has_pos.notify_all()
        return result

    def put(self, data):
        """
        Put the data into this buffer. The current thread is blocked if the buffer is full.
        :param data:
        :return: the data
        """
        with self.has_pos:
            while len(self.buffer) >= self.size:
                self.has_pos.wait()
            self.buffer.append(data)
            self.has_data.notify_all()


if __name__ == '__main__':
    import time
    buffer = Buffer(10)
    def get():
        for _ in range(10000):
            print(buffer.get())
            time.sleep(0.01)
    def put():
        for i in range(10000):
            buffer.put(i)

    th_put = threading.Thread(target=put, daemon=True)
    th_get = threading.Thread(target=get, daemon=True)
    th_put.start()
    th_get.start()
    th_put.join()
    th_get.join()

