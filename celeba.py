#coding=utf-8
import zipfile
import numpy as np
import cv2
import os
import numpy as np
import threading

class CelebA:
    def __init__(self, path_img, path_ann, path_bbox):
        self._process_imgs(path_img)
        self._process_anns(path_ann)
        self._process_bboxes(path_bbox)


        self.pos = np.random.randint(0, self.num_examples)
        self.pos1 = np.random.randint(0, self.num_examples)
        self.pos2 = np.random.randint(0, self.num_examples)
        self.persons = max(self.ids) + 1
        print('Persons =', self.persons)
        assert self.persons == len(set(self.ids))

    def pick_2_samples(self):
        # for i in range(2):
        filename1 = self.filenames[self.pos1]
        img1 = self.zf.read(filename1)
        img1 = np.frombuffer(img1, np.uint8)
        img1 = cv2.imdecode(img1, 1)
        # cv2.imshow('my_img', img)
        # cv2.imwrite('../samples/photos/sample_%d' % i, img)
        cv2.imencode('.jpg', img1)[1].tofile('../samples/photos/sample_1.jpg')

        filename2 = self.filenames[self.pos2]
        img2 = self.zf.read(filename2)
        img2 = np.frombuffer(img2, np.uint8)
        img2 = cv2.imdecode(img2, 1)
        # cv2.imshow('my_img', img)
        # cv2.imwrite('../samples/photos/sample_%d' % i, img)
        cv2.imencode('.jpg', img2)[1].tofile('../samples/photos/sample_2.jpg')

    def _process_bboxes(self, path_bbox):
        self.bboxes = []
        with open(path_bbox) as file:
            lines = file.readlines()
            del lines[0]
            del lines[0]
            for line in lines:
                locs = []
                for loc in line.split(' '):
                    if len(loc) > 0:
                        locs.append(loc)
                del locs[0]
                locs = [int(e) for e in locs]
                self.bboxes.append(locs)
        # print(self.bboxes)

    def _process_anns(self, path_ann):
        with open(path_ann) as file:
            lines = file.readlines()
        self.ids = []
        for line in lines:
            id = int(line[line.find(' ') + 1:]) - 1
            self.ids.append(id)

    def _process_imgs(self, path_img):
        self.filenames = []
        self.zf = zipfile.ZipFile(path_img)
        # print(self.zf.filelist)
        for info in self.zf.filelist:
            # if info.is_dir(): continue
            if info == self.zf.filelist[0]: continue
            self.filenames.append(info.filename)
            if len(self.filenames) % 10000 == 0:
                print('read %d images' % len(self.filenames))
        print('Read %d images from %s successfully' % (len(self.filenames), path_img), flush=True)

    @property
    def num_examples(self):
        return len(self.filenames)

    def next_batch(self, batch_size):
        next = self.pos + batch_size
        num = self.num_examples
        if next < num:
            result = self.filenames[self.pos: next], self.ids[self.pos: next], self.bboxes[self.pos: next]
        else:
            result = self.filenames[self.pos: num], self.ids[self.pos: num], self.bboxes[self.pos: num]
            next -= num
            result = result[0] + self.filenames[:next], result[1] + self.ids[:next], result[2] + self.bboxes[:next]
        self.pos = next
        res = []
        for filename, (x, y, w, h) in zip(result[0], result[2]):
            img = self.zf.read(filename)
            img = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(img, 1)
            # img = img[x: x+w][y, y+h]
            res.append(img)
        return res, result[1]

    def close(self):
        self.zf.close()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    # path_img = os.path.join(os.getcwd(), 'samples\celeba\Img\img_align_celeba.zip')
    # path_ann = os.path.join(os.getcwd(), 'samples\celeba\Anno\identity_CelebA.txt')
    # path_bbox = os.path.join(os.getcwd(), 'samples\celeba\Anno\list_bbox_celeba.txt')
    path_img = '../samples/celeba/Img/img_align_celeba.zip'
    path_ann = '../samples/celeba/Anno/identity_CelebA.txt'
    path_bbox = '../samples/celeba/Anno/list_bbox_celeba.txt'
    path_sample = '../samples/photos/'
    ca = CelebA(path_img, path_ann, path_bbox)
    # ca.pick_2_samples()
    with ca:
        for _ in range(1):
            imgs, _ = ca.next_batch(100)
            cv2.imshow('My img', imgs[0])
            cv2.waitKey()