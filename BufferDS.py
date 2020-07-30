from buffer import Buffer
import threading


class BufferDS:
    def __init__(self, buffer_size, ds, batch_size):
        # ds.num_examples, ds.next_batch(batch_size)
        self.ds = ds
        self.batch_size = batch_size
        self.buffer = Buffer(buffer_size)
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    @property
    def num_examples(self):
        return self.ds.num_examples

    def next_batch(self, batch_size):
        return self.buffer.get()

    def read(self):
        while True:
            data = self.ds.next_batch(self.batch_size)
            self.buffer.put(data)
