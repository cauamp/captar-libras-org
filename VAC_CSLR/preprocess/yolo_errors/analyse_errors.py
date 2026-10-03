import os
import cv2
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', './include'))

from video_detection import detection_batch


def convert_avi_to_mp4_cv(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"File {input_file} does not exist.")
        return False
    # Open the input video file
    cap = cv2.VideoCapture(input_file)

    # Get the width, height, and frames per second (fps) of the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if cap is None or not cap.isOpened():
        print("Error opening video stream or file")
        return False
    
    frames_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if np.all(frame <= 20):
            print(f"Frame {len(frames_list)} is all black")
            
        out.write(frame)
        frames_list.append(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return frames_list

def save_yolo_fails(frames_list, output_dir):
    errors = []
    boxes = detection_batch(frames_list)
    for i, box in enumerate(boxes):
        if None in box:
            print(f"Frame {i} has no person detected")
            errors.append(i)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(f"{output_dir}/{i}.jpg", frames_list[i])
            if i > 0:	
                cv2.imwrite(f"{output_dir}/{i-1}.jpg", frames_list[i-1])
            if i < len(frames_list)-1:
                cv2.imwrite(f"{output_dir}/{i+1}.jpg", frames_list[i+1])     
    return errors       
if __name__ == "__main__":
    df = pd.read_csv("../errors.csv")
    ids = df[df['Erro'] == 'NONE IN CROP']['VideoID'].tolist()
    #ids = random.sample(ids, 50)

    df_train = pd.read_csv("../../datasets_/2024-10-16-checked/train.csv")
    df_test = pd.read_csv("../../datasets_/2024-10-16-checked/test.csv")
    df_validate = pd.read_csv("../../datasets_/2024-10-16-checked/validate.csv")
    df = pd.concat([df_train, df_test, df_validate])
    
    paths = []
    paths = (df[df['video_id'].isin(ids)]['video_path'])
    paths = paths.tolist()
 
    output_dir = './out'
    os.makedirs(output_dir, exist_ok=True)
    for file_path in paths:
        print(f'Processing {file_path.split("/")[-1]}')
        input_file = file_path.replace('.mp4', '.avi')
        output_file = os.path.join(
            output_dir, input_file.split("/")[-1].replace(".avi", ".mp4")
        )
        
        frames_list = convert_avi_to_mp4_cv(input_file, output_file)
        if not frames_list:
            continue
        
        out_dir = output_file.replace('.mp4', '')
        print(f"Checking for yolo fails in {input_file}")
        failed_frames = save_yolo_fails(frames_list, out_dir)
        if failed_frames == [] or failed_frames[-1] < 3:
            os.remove(output_file)
        else:
            print(f"Failed frames: {failed_frames}")
                
