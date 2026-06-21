import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from sklearn.decomposition import PCA
from tqdm import tqdm


import os

def flows_from_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        cap.release()
        exit()
    # Initial Frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        cap.release()
        exit()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow_matrices = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break 
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_matrices.append(flow)
        prev_gray = next_gray
    cap.release()
    return flow_matrices


# lecture_1_flows = flows_from_video("D:\ASH\datasets\Le2i\Coffee_room_all\Videos\\video (1).avi") #TYPE PATH TO VIDEO OF CHOICE
# print(np.shape(lecture_1_flows))

# lecture_1_flows_array = np.array(lecture_1_flows)

# num_frames, height, width, num_components = lecture_1_flows_array.shape

# # Flatten the spatial dimensions and keep the flow components together
# optical_flow_flat = lecture_1_flows_array.reshape(num_frames, height * width * num_components)

# # Initialize PCA
# pca = PCA(n_components=64) 

# reduced_flows = pca.fit_transform(optical_flow_flat)
# print(reduced_flows.shape)

# Specify the folder path
folder_path = "D:\ASH\datasets\Le2i\Office\Videos"
dst_folder = "D:\ASH\datasets\Le2i\Office\Flows"
for filename in tqdm(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        flows = flows_from_video(file_path)
        flows = np.array(flows)
        num_frames, height, width, num_components = flows.shape
        optical_flow_flat = flows.reshape(num_frames, height * width * num_components)
        pca = PCA(n_components=64) 
        reduced_flows = pca.fit_transform(optical_flow_flat)
        np.save(os.path.join(dst_folder, filename[:-4]), reduced_flows)