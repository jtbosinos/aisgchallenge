from argparse import ArgumentParser
import pandas as pd
import os
import cv2 
import numpy as np
from decord import VideoReader
from decord import cpu
import pandas as pd
from skimage.io import imread
import moviepy.editor as mp
from tqdm import tqdm
import gc
import random
import joblib as joblib
import lightgbm as lgb

photo_dir = "./frames"

img_width = 180
img_height = 320
    
def process_img(image, width, height, rotate = True):
    #Check if image is in landscape mode then rotate to make it portrait
    if ((image.shape[0] < image.shape[1]) and (rotate == True)) :
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        dim = (width, height)
    elif ((image.shape[0] < image.shape[1]) and (rotate == False)):
        dim = (height, width)
    else:
        dim = (width, height)
        
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    # load the VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)
                     
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0

    if every > 25 and len(frames_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = vr.get_batch(frames_list).asnumpy()

        for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(index))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # save the extracted image
                saved_count += 1  # increment our counter by one

    else:  # this is faster for every <25 and consumes small memory
        for index in range(start, end):  # lets loop through the frames until the end
            frame = vr[index]  # read an image from the capture
            
            if index % every == 0:  # if this is a frame we want to write out based on the 'every' argument
                save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(index))  # create the save path
                if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                    cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save the extracted image
                    saved_count += 1  # increment our counter by one

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=1):
    """
    Extracts the frames from a video
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)
    
    # print("Extracting frames from {}".format(video_filename))
    
    extract_frames(video_path, frames_dir, every=every)  # let's now extract the frames

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames
    

def main(input_dir, output_file):
    # read input directory for mp4 videos only
    # note: all files would be mp4 videos in the mounted input directory

    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)    

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".mp4") or filename.endswith(".mov"):
            video_to_frames(video_path=os.path.join(input_dir, filename), frames_dir=photo_dir, overwrite=True, every=60)
            # print(os.path.join(video_dir, filename))
            continue
        else:
            continue
        
  
    df_grayscale = pd.DataFrame()

    for filedir in tqdm(os.listdir(photo_dir)):
        randomnums = random.sample(range(0, len(os.listdir(os.path.join(photo_dir, filedir)))), 5 if len(os.listdir(os.path.join(photo_dir, filedir))) >= 5 else len(os.listdir(os.path.join(photo_dir, filedir))))
        ctr = -1
        for filename in os.listdir(os.path.join(photo_dir, filedir)):
            ctr = ctr + 1
            if ctr in randomnums:
                if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                    image = imread(os.path.join(photo_dir, filedir, filename), as_gray=True) 
                    image = process_img(image, img_width, img_height)

                    features = np.reshape(image, (image.shape[0]*image.shape[1]))
                    df_tmp = pd.DataFrame(features.reshape(-1, len(features)))
                    df_tmp['vid_filename'] = filedir
                    df_tmp['img_filename'] = filename
                    df_grayscale = df_grayscale.append(df_tmp)
                    continue
                else:
                    continue
                    
    df_vidtype = pd.DataFrame()

    for filedir in tqdm(os.listdir(photo_dir)):
        for filename in os.listdir(os.path.join(photo_dir, filedir)):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                image = imread(os.path.join(photo_dir, filedir, filename), as_gray=True) 
                portrait = 1
                if image.shape[0] < image.shape[1]:
                    portrait = 0

                df_tmp = pd.DataFrame([[portrait]])
                df_tmp.columns = ['portrait_f']
                df_tmp['vid_filename'] = filedir
                df_tmp['img_filename'] = filename
                df_vidtype = df_vidtype.append(df_tmp)
                continue
            else:
                continue

    data = df_grayscale.merge(df_vidtype,how='left',left_on=['vid_filename','img_filename'],right_on=['vid_filename','img_filename'])
    dataOrig = data.loc[:,['vid_filename','img_filename']]
    data = data.drop(['vid_filename','img_filename'], axis = 1)
    
    lgbm = joblib.load("./lgbm_1.pkl")
    
    predictions = pd.DataFrame(lgbm.predict_proba(data), columns=["prob_0", "prob_1"]).set_index(data.index)
    predictions = pd.concat([dataOrig, predictions], axis=1)
    
    ks_df = joblib.load("./ks_df_1.pkl")
    min_prob = ks_df.loc[ks_df['ks']==ks_df.ks.max()].index.values
    threshold = ks_df.iloc[min_prob, 0]
    
    output = predictions.groupby(['vid_filename'], as_index=False)['prob_1'].mean()
    output = output.rename(columns={'vid_filename': 'filename', 'prob_1': 'probability'})
    output.to_csv(output_file, sep=',', encoding='utf-8', index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-input", type=str, required=True, help="Input directory of test videos")
    parser.add_argument("-output", type=str, required=True, help="Output directory with filename e.g. /data/output/submission.csv")
    args = parser.parse_args()

    main(input_dir=args.input, output_file=args.output)