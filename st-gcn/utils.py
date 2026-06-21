import os
import cv2
import numpy as np

def clean(dataset, threshold=5, verbose=False):
    cleaned = []
    for data in dataset:
        frame_ids = data['frame_ids']
        count = 0
        for i in range(1, len(frame_ids)):
            count += frame_ids[i] - frame_ids[i - 1] - 1
        if count > threshold:
            if verbose: print(data['filename'])
        else:
            cleaned.append(data)
    return np.array(cleaned)

def visualize(data, video_path):
    """Visualize video with AlphaPose bounding box

    Args:
        data (list): AlphaPose result read from a JSON file
        video_path (str): path to the video
    
    # example of generating 'data'
    file_path = "D:\ASH\datasets\Le2i\Coffee_room_all\Skeletons\\video (1).json"
    with open(file_path, 'r') as json_file:
        data_str = json_file.read()
    data = eval(data_str)
    """
    # Initialize video capture from a video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Loop through each frame of the video
    f_idx = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            break

        if f_idx >= 45 and f_idx <= 80: 
            if f_idx - 9 >= 0:
                xmin, ymin, width, height = map(int, data[f_idx]['box'])
                # Draw the rectangle on the image
                cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
            cv2.imshow('Video with Bounding Boxes', frame)
        else: 
            height, width, _ = frame.shape
            cv2.imshow('Video with Bounding Boxes', np.zeros((height, width, 3), dtype=np.uint8))
            
        f_idx += 1

        # Wait for a key press and check if it's the 'q' key to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def split_fall_seq(filename, seq, start, end, nframes, shift_window=0):
    # seq.shape == (num_frames, 17, 2)
    frame_len = nframes + 2 * shift_window
    pivot = start
    
    left = []
    start_frames_left = []
    for i in range(0, pivot, nframes):
        end_frame = min(pivot, i + nframes, len(seq))
        start_frame = i
        if end_frame - start_frame != nframes:
            start_frame -= nframes - (end_frame - start_frame)
            if start_frame < 0:
                offset = 0 - start_frame
                start_frame += offset
                end_frame += offset
                
        adjusted_start_frame = max(start_frame - shift_window, 0)
        adjusted_end_frame = min(end_frame + shift_window, len(seq))
        pad_start = start_frame - adjusted_start_frame
        pad_end = adjusted_end_frame - end_frame
        if pad_start < shift_window:
            adjusted_end_frame += shift_window - pad_start
        elif pad_end < shift_window:
            adjusted_start_frame -= shift_window - pad_end
        
        subseq = seq[adjusted_start_frame:adjusted_end_frame]
        if len(subseq) != frame_len:
            print(start_frame, end_frame, len(seq), pivot, start, filename)
            print(len(subseq))
        left.append(subseq)
        start_frames_left.append(start_frame)
    
    right = []
    start_frames_right = []
    for i in range(pivot + nframes, len(seq), nframes):
        end_frame = min(len(seq), i + nframes)
        start_frame = i
        if end_frame - start_frame != nframes:
            start_frame -= nframes - (end_frame - start_frame)
            if start_frame < 0:
                offset = 0 - start_frame
                start_frame += offset
                end_frame += offset
                
        adjusted_start_frame = max(start_frame - shift_window, 0)
        adjusted_end_frame = min(end_frame + shift_window, len(seq))
        pad_start = start_frame - adjusted_start_frame
        pad_end = adjusted_end_frame - end_frame
        if pad_start < shift_window:
            adjusted_end_frame += shift_window - pad_start
        elif pad_end < shift_window:
            adjusted_start_frame -= shift_window - pad_end
        
        subseq = seq[adjusted_start_frame:adjusted_end_frame]
        right.append(subseq)
        start_frames_right.append(start_frame)
    
    end_frame = min(len(seq), start + nframes)
    start_frame = start
    if end_frame - start_frame != nframes:
        start_frame -= nframes - (end_frame - start_frame)
        if start_frame < 0:
            offset = 0 - start_frame
            start_frame += offset
            end_frame += offset
    adjusted_start_frame = max(start_frame - shift_window, 0)
    adjusted_end_frame = min(end_frame + shift_window, len(seq))
    pad_start = start_frame - adjusted_start_frame
    pad_end = adjusted_end_frame - end_frame
    if pad_start < shift_window:
        adjusted_end_frame += shift_window - pad_start
    elif pad_end < shift_window:
        adjusted_start_frame -= shift_window - pad_end
    
    fall_seq = seq[adjusted_start_frame:adjusted_end_frame]
    # fall_seq = seq[start_frame:end_frame] # the subseq of the fall event
    
    non_falls = left + right # a list of subseqs of all the non-fall events
    start_frames = start_frames_left + start_frames_right # starting frame of each non fall events
    
    return fall_seq, non_falls, start_frames, start_frame

def split_seq(seq, nframes, shift_window=0):
    res = []
    start_frames = []
    for i in range(0, len(seq), nframes):
        end_frame = min(len(seq), i + nframes)
        start_frame = i
        if end_frame - start_frame != nframes:
            start_frame -= nframes - (end_frame - start_frame)
            if start_frame < 0:
                offset = 0 - start_frame
                start_frame += offset
                end_frame += offset
        adjusted_start_frame = max(start_frame - shift_window, 0)
        adjusted_end_frame = min(end_frame + shift_window, len(seq))
        pad_start = start_frame - adjusted_start_frame
        pad_end = adjusted_end_frame - end_frame
        if pad_start < shift_window:
            adjusted_end_frame += shift_window - pad_start
        elif pad_end < shift_window:
            adjusted_start_frame -= shift_window - pad_end
        subseq = seq[adjusted_start_frame:adjusted_end_frame]
        res.append(subseq)
        start_frames.append(start_frame)
    
    return res, start_frames

def add_flows(flows, subseqs, start_frames, frame_ids, nframes):
    new_subseqs = []
    for i, subseq in enumerate(subseqs):
        start = start_frames[i]
        new_subseq = dict()
        new_subseq['skeletons'] = subseq.copy()
        new_subseq['flows'] = []
        
        for j in range(nframes):
            if frame_ids[start + j] < len(flows):
                new_subseq['flows'].append(flows[frame_ids[start + j]])
            else:
                new_subseq['flows'].append(flows[-1])
            
        new_subseq['flows'] = np.array(new_subseq['flows'])
        new_subseqs.append(new_subseq)
    return new_subseqs
    
def split_skeletons(skeletons, nframes=45, shift_window=0, flow_path=None):
    # skeletons is an array of preprocessed skeleton dicts with keys filename, keypoints, scores, boxes, fall_interval, offset
    res_falls = []
    res_non_falls = []
    for seq in skeletons:
        filename = seq['filename']
        
        # start and end frame of the fall event in the skeleton sequence
        offset = seq['offset']
        start_frame, end_frame = seq['fall_interval']
        
        if start_frame == 0 and end_frame == 0:
            non_fall, start_frames = split_seq(seq['keypoints'], nframes, shift_window)
        else:
            fall, non_fall, start_frames, fall_start_frame = split_fall_seq(filename, seq['keypoints'], start_frame, end_frame, nframes, shift_window)
            res_falls.append(fall)
            
        res_non_falls += non_fall
    
    return res_falls, res_non_falls

def split_skeletons_and_flows(skeletons, nframes=45):
    # skeletons is an array of preprocessed skeleton dicts with keys filename, keypoints, scores, boxes, fall_interval, offset
    res_falls = []
    res_non_falls = []
    for seq in skeletons:
        filename = seq['filename']
        
        # start and end frame of the fall event in the skeleton sequence
        offset = seq['offset']
        start_frame, end_frame = seq['fall_interval']
        
        if start_frame == 0 and end_frame == 0:
            non_fall, start_frames = split_seq(seq['keypoints'], nframes)
        else:
            fall, non_fall, start_frames, fall_start_frame = split_fall_seq(filename, seq['keypoints'], start_frame, end_frame, nframes)
            fall = add_flows(seq['flows'], [fall], [fall_start_frame], seq['frame_ids'], nframes)[0]
            res_falls.append(fall)
        
        non_fall = add_flows(seq['flows'], non_fall, start_frames, seq['frame_ids'], nframes)
            
        res_non_falls += non_fall
    
    return res_falls, res_non_falls