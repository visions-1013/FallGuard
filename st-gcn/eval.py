import os
import re
import torch
import argparse
import numpy as np

from model.stgcn import stgcn, flow_stgcn
from model.st_graph import get_distance_adjacency
from tqdm import tqdm

HOME_PATH = "D:\ASH\datasets\Le2i\Home_all\Skeletons_full"
OFFICE_PATH = "D:\ASH\datasets\Le2i\Office\Skeletons_split"
LEC_ROOM_PATH = "D:\ASH\datasets\Le2i\Lecture room\Skeletons_split"

HOME_LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
OFFICE_LABELS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
LEC_ROOM_LABELS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

default_layer_config = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
                        (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)] # (in_channles, out_channels, temporal_stride)
layer_config = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
                (128, 128, 1), (128, 256, 2), (256, 256, 1)]

edges = [
    [0, 1],  # Nose - Left Eye
    [0, 2],  # Nose - Right Eye
    [1, 3],  # Left Eye - Left Ear
    [2, 4],  # Right Eye - Right Ear
    [1, 5],  # Left Eye - Left Shoulder
    [2, 6],  # Right Eye - Right Shoulder
    [5, 7],  # Left Shoulder - Left Elbow
    [6, 8],  # Right Shoulder - Right Elbow
    [7, 9],  # Left Elbow - Left Wrist
    [8, 10],  # Right Elbow - Right Wrist
    [5, 11],  # Left Shoulder - Left Hip
    [6, 12],  # Right Shoulder - Right Hip
    [11, 13],  # Left Hip - Left Knee
    [12, 14],  # Right Hip - Right Knee
    [13, 15],  # Left Knee - Left Ankle
    [14, 16]   # Right Knee - Right Ankle
]

max_x = 320
max_y = 240

def numerical_sort(value):
    """
    This helper function finds numbers in a string and returns the string parts and numerical parts.
    """
    parts = re.split('(\d+)', value)
    parts[1::2] = map(int, parts[1::2])  # Convert numerical strings to integers
    return parts

def load_data(path, flow_path=None):
    data = []
    files = sorted(os.listdir(path), key=numerical_sort)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            seq = np.load(file_path, allow_pickle=True).item()
    
            if flow_path:
                flow_file_path = os.path.join(flow_path, file)
                if os.path.isfile(flow_file_path):
                    flows = np.load(flow_file_path)
                    data.append((seq['filename'], seq['keypoints'], flows, seq['frame_ids']))
            else:
                data.append((seq['filename'], seq['keypoints'], seq['frame_ids']))

    return data

def create_batch(frames, stride=1, segment_length=45, in_channels=2):
    batch = []
    start = 0
    while start + segment_length <= len(frames):
        end = start + segment_length
        batch.append(frames[start:end, :, :])
        start += stride
    
    if start < len(frames):
        end = len(frames)
        start = end - segment_length
        batch.append(frames[start:end, :, :])

    batch = np.array(batch)
    if in_channels == 3:
        batch = batch.astype(np.float32)
    else:
        batch = batch[:, :, :, :2].astype(np.float32)
    batch[:, :, :, 0] /= max_x
    batch[:, :, :, 1] /= max_y
    batch = np.transpose(batch, (0, 3, 1, 2))
    
    return batch

def create_batch_flow(frames, flows, frame_ids, stride=1, segment_length=45):
    flows = np.vstack([flows, flows[-1]])
    skeleton_batch = []
    flow_batch = []
    start = 0
    while start + segment_length <= len(frames):
        end = start + segment_length
        skeleton_batch.append(frames[start:end, :, :])
        flow_batch.append(flows[frame_ids[start:end]])
        start += stride
    
    if start < len(frames):
        end = len(frames)
        start = end - segment_length
        skeleton_batch.append(frames[start:end, :, :])
        flow_batch.append(flows[frame_ids[start:end]])

    skeleton_batch = np.array(skeleton_batch)
    skeleton_batch = skeleton_batch[:, :, :, :2].astype(np.float32)
    skeleton_batch[:, :, :, 0] /= max_x
    skeleton_batch[:, :, :, 1] /= max_y
    skeleton_batch = np.transpose(skeleton_batch, (0, 3, 1, 2))
    
    flow_batch = np.array(flow_batch)
    
    flow_batch = np.transpose(flow_batch, (0, 2, 1))
    
    return skeleton_batch, flow_batch

def evaluate(model, device, stride=1, testset='lecture room', profile=False, in_channels=2):
    model.eval()
    
    # 'data' is a list of tuples: (filename, keypoints)
    if not isinstance(testset, str):
        data, labels = testset
    elif testset == 'home':
        data = load_data(HOME_PATH)
        labels = HOME_LABELS
    elif testset == 'office':
        data = load_data(OFFICE_PATH)
        labels = OFFICE_LABELS
    else:
        data = load_data(LEC_ROOM_PATH)
        labels = LEC_ROOM_LABELS
    
    acc = 0
    tp = []
    tn = []
    fp = []
    fn = []
    results = []
    for i in tqdm(range(len(data))):
        filename, frames, frame_ids = data[i] # (num_frames, 17, 2)
        
        batch = create_batch(frames, stride=stride, in_channels=in_channels)
        batch = torch.tensor(batch).to(device)
        
        with torch.no_grad():
            y_pred = model(batch)
            if labels[i] == 1:
                y_batch = torch.ones(len(batch)).to(device)
                res = torch.argmax(y_pred, axis=-1)
                
                if torch.any(res == y_batch):
                    pred = 1
                    acc += 1
                    tp.append(filename)
                else:
                    pred = 0
                    fn.append(filename)
                
                if profile: 
                    results.append((filename, 1, pred, res.detach().cpu(), y_pred[torch.arange(y_pred.size(0)), res].detach().cpu()))
            else:
                y_batch = torch.zeros(len(batch)).to(device)
                res = torch.argmax(y_pred, axis=-1)
                
                if torch.all(res == y_batch):
                    pred = 0
                    acc += 1
                    tn.append(filename)
                else:
                    pred = 1
                    fp.append(filename)
                
                if profile: 
                    results.append((filename, 0, pred, res.detach().cpu(), y_pred[torch.arange(y_pred.size(0)), res].detach().cpu()))
    
    print("True positives:", tp)
    print("True negatives:", tn)     
    print("False positives:", fp)
    print("False negatives:", fn)
    
    accuracy = acc / len(data)
    specificity = len(tn) / (len(tn) + len(fp))
    sensitivity = len(tp) / (len(tp) + len(fn))
    return accuracy, specificity, sensitivity, results

def evaluate_flow(model, device, stride=1, testset='lecture room', profile=False):
    model.eval()
    
    data, labels = testset
    
    acc = 0
    fp = []
    fn = []
    results = []
    for i in tqdm(range(len(data))):
        filename, skeletons, flows, frame_ids = data[i]
        
        skeleton_batch, flow_batch = create_batch_flow(skeletons, flows, frame_ids, stride=stride)
        skeleton_batch = torch.tensor(skeleton_batch).to(device)
        flow_batch = torch.tensor(flow_batch).to(device)
        
        with torch.no_grad():
            y_pred = model(skeleton_batch, flow_batch)
            if labels[i] == 1:
                y_batch = torch.ones(len(skeleton_batch)).to(device)
                res = torch.argmax(y_pred, axis=-1)
                
                if torch.any(res == y_batch):
                    pred = 1
                    acc += 1
                else:
                    pred = 0
                    fn.append(filename)
                
                if profile: 
                    results.append((filename, 1, pred, res.detach().cpu(), y_pred[torch.arange(y_pred.size(0)), res].detach().cpu()))
            else:
                y_batch = torch.zeros(len(skeleton_batch)).to(device)
                res = torch.argmax(y_pred, axis=-1)
                
                if torch.all(res == y_batch):
                    pred = 0
                    acc += 1
                else:
                    pred = 1
                    fp.append(filename)
                
                if profile: 
                    results.append((filename, 0, pred, res.detach().cpu(), y_pred[torch.arange(y_pred.size(0)), res].detach().cpu()))
            
    print("False positives:", fp)
    print("False negatives:", fn)
    print("Accuracy:", acc / len(data))
    return acc / len(data), results
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True,
                    help='path to the model weights')
    parser.add_argument('--testset', type=str, default='lecture room',
                    help='which test set to use, can be office or lecture room')
    parser.add_argument('--dst', type=str, default='./',
                    help='path to the destination folder')
    
    args = parser.parse_args()
        
    num_node = 17
    A = get_distance_adjacency(np.array(edges), num_node)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = stgcn(num_class=2, window_size=45, num_point=17, graph=A, layer_config=default_layer_config)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    print(f"Evaluating on set: {args.testset}")
    accuracy = evaluate(model, device, stride=1, testset=args.testset)
    print(accuracy)