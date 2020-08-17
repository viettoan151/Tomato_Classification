import cv2
import argparse
import numpy as np
import os
import time

from yolo_opencv_inference import Yolo_OpenCV, NM_suppression, draw_bounding_box

WORK_MODE = {
    'camera':0,
    'video':1,
    'image':2,
}

# handle command line arguments
ap = argparse.ArgumentParser()

ap.add_argument('-m', '--mode', type=int,
                default='0',
                help=WORK_MODE)
ap.add_argument('-i', '--input',
                default = './Image_demo/01_ts/',
                help = 'path to input image')
ap.add_argument('-o', '--output',
                default = './Image_demo/01_ts_out/',
                help = 'path to out image')

ap.add_argument('-c', '--config',
                default = './cfg/Combros_tomato_yolov2_tiny.cfg',
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights',
                default = './weights/Combros_tomato_yolov2_tiny_6000.weights',
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes',
                default = './cfg/classes_tomato.txt',
                help = 'path to text file containing class names')

args = ap.parse_args()

def video_inference(in_file, out_file, yolo_network):
    # initialize the video stream and allow the camera sensor to warmup
    video_in = cv2.VideoCapture(in_file)
    
    f_rate = video_in.get(cv2.CAP_PROP_FPS)
    f_width = video_in.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = video_in.get(cv2.CAP_PROP_FRAME_HEIGHT)
    f_count = video_in.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Write the video
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    video_out = cv2.VideoWriter('fast_yolov2_tomato_test_2.avi', fourcc, f_rate, (f_width, f_height))

    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(yolo_network.yolo_classes), 3))
    
    read_cnt = 0
    processing_time = []
    while True:
        # grab the frame from the file
        (grabbed, frame) = video_in.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        
        time_start = time.time()
        output = yolo_network.inference(frame)
        output = NM_suppression(output, 0.5, 0.4)
        
        # go through the detections remaining
        # after nms and draw bounding box
        image = frame.copy()
        for idx, cls in enumerate(output['classese']):
            info = {'id': cls,
                    'name': yolo_network.yolo_classes[cls],
                    'cfd': output['confidences'][idx]}
            box = output['bboxes'][idx]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_bounding_box(image, [(round(x), round(y)), (round(x+w), round(y+h))], COLORS[cls], info)
    
        ptime = time.time() - time_start
        processing_time.append(ptime)
        print('frame:', read_cnt)
        read_cnt +=1
        
        video_out.write(image)
        
        # wait until any key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release resources
    video_in.release()
    video_out.release()
    print('Average inference time:', np.mean(processing_time))
    print('End of Video mode')

if __name__=='__main__':
    print('Hello, this is tomato classification')
    
    # read pre-trained model and config file
    yolo_net = Yolo_OpenCV(args.weights, args.config)
    
    if args.mode == WORK_MODE['camera']:
        print('Camera mode')
        
    elif args.mode == WORK_MODE['video']:
        print('Video mode')
        video_inference(args.input, args.output, yolo_net)
        
    elif args.mode == WORK_MODE['image']:
        print('Image mode')









