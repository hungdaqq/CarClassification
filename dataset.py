import numpy as np
import os
import re
import pickle
import cv2 as cv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

class Labels: 
    def __init__(self, file_path):
        self.file_path = file_path
    def load_pickle(self):
        self.class_name = pickle.loads(open(self.file_path, "rb").read())
        self.class_list = np.array(self.class_name).tolist()
        return self.class_list
    def write_pickle(self, class_name):
        self.class_name = class_name
        f = open(self.file_path, "wb")
        f.write(pickle.dumps(self.class_name))
        f.close()

class Dataset:
    def __init__(self, folder_path, subset):
        self.folder_path = folder_path
        self.subset = subset
        self.path_to_subset = self.folder_path + self.subset
    def dataset_collect(self, input_shape = (224,224,3)):
        X = []
        Y = []
        for folder in os.listdir(self.path_to_subset):
            for image in os.listdir(os.path.join(self.path_to_subset, folder)):
                path_to_image = os.path.join(self.path_to_subset, folder, image)
                image = cv.imread(path_to_image)
                image = cv.resize(image, (input_shape[1], input_shape[0]))
                label = re.findall(r'\w+\_\w+\_\w+', path_to_image)[0].split('_')
                X.append(image)
                Y.append(label)
        return np.array(X), np.array(Y)      
      
    def dataset_health(self):
        self.health_dict = {}
        labels = Labels('./working/class_name.pickle')
        class_list = labels.load_pickle()
        for i in range(len(class_list)):
            self.health_dict[class_list[i]] = 0    
        for folder in os.listdir(self.path_to_subset):
            for image in os.listdir(os.path.join(self.path_to_subset, folder)):
                path_to_image = os.path.join(self.path_to_subset, folder, image)
                label = re.findall(r'\w+\_\w+\_\w+', path_to_image)[0].split('_')
                for i in range(len(class_list)):
                    if label[0] == class_list[i] or label[1] == class_list[i] or label[2] == class_list[i]:
                        self.health_dict[class_list[i]] +=1
        return self.health_dict   

    def dataset_split(self, X, Y, train_size, test_size):
        train_images, temp_images, train_labels, temp_labels = train_test_split(X, Y, test_size=1-train_size, random_state=42)
        val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=test_size, random_state=42)
        return train_images, train_labels, val_images, val_labels, test_images, test_labels 

# health = Dataset('./dataset/')
# print(health.dataset_collect())
# def rename(name):
#     new_name = name.lower().capitalize()
#     return new_name

# for subset in ('train', 'test'):
#     path_to_subset = f'/home/hung/Documents/GitHub/CarClassification/dataset/{subset}/'
#     os.chdir(path_to_subset)
#     # Loop through all classes in subset
#     for folder in os.listdir(path_to_subset):
#         label = re.findall(r'\w+\_\w+\_\w+', folder)[0].split('_')
#         new_name = 'brand' + rename(label[0]) + '_' + 'category' + rename(label[1]) + '_' + 'color' + rename(label[2])
#         os.rename(path_to_subset+folder, path_to_subset+new_name)
