import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np


class ApexObstacleDataset(torch.utils.data.Dataset):
    def __init__(self, directory, categories, transform=None, random_hflip=False):
        super(ApexObstacleDataset, self).__init__()
        self.directory = directory
        self.categories = categories
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        # normalize all points to [-1, 1]
        def norm_pt(pt_x, pt_y):
            return 2.0 * (pt_x / width - 0.5), 2.0 * (pt_y / height - 0.5)
            
        ax, ay = norm_pt(ann['ax'], ann['ay'])
        x1, y1 = norm_pt(ann['x1'], ann['y1'])
        x2, y2 = norm_pt(ann['x2'], ann['y2'])
        x3, y3 = norm_pt(ann['x3'], ann['y3'])
        x4, y4 = norm_pt(ann['x4'], ann['y4'])
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            ax = -ax
            x1 = -x1
            x2 = -x2
            x3 = -x3
            x4 = -x4
            
        return image, ann['category_index'], torch.Tensor([ax, ay, x1, y1, x2, y2, x3, y3, x4, y4])
    
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        # format: ax_ay_x1_y1_x2_y2_x3_y3_x4_y4_uuid.jpg
        return int(items[0]), int(items[1]), int(items[2]), int(items[3]), int(items[4]), int(items[5]), int(items[6]), int(items[7]), int(items[8]), int(items[9])
        
    def refresh(self):
        self.annotations = []
        for category in self.categories:
            category_index = self.categories.index(category)
            for image_path in glob.glob(os.path.join(self.directory, category, '*.jpg')):
                try:
                    ax, ay, x1, y1, x2, y2, x3, y3, x4, y4 = self._parse(image_path)
                    self.annotations += [{
                        'image_path': image_path,
                        'category_index': category_index,
                        'category': category,
                        'ax': ax, 'ay': ay,
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'x3': x3, 'y3': y3,
                        'x4': x4, 'y4': y4
                    }]
                except Exception as e:
                    pass
                    
    def save_entry(self, category, image, pts):
        # pts is a list of 5 tuples: [(ax,ay), (x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        category_dir = os.path.join(self.directory, category)
        if not os.path.exists(category_dir):
            subprocess.call(['mkdir', '-p', category_dir])
            
        ax, ay = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        x3, y3 = pts[3]
        x4, y4 = pts[4]
        
        filename = '%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%s.jpg' % (ax, ay, x1, y1, x2, y2, x3, y3, x4, y4, str(uuid.uuid1()))
        
        image_path = os.path.join(category_dir, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
        
    def get_count(self, category):
        i = 0
        for a in self.annotations:
            if a['category'] == category:
                i += 1
        return i
