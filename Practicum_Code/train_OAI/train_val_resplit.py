import numpy as np
import glob
import argparse
import random
import shutil
import csv
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default="/project/o/oespinga/shiyuxin/OAI_images/1205728_P001_yolov8/train")
    parser.add_argument('--val_path', default="/project/o/oespinga/shiyuxin/OAI_images/1205728_P001_yolov8/val")
    parser.add_argument('--matchbook', default="/project/o/oespinga/shiyuxin/OAI_images/OAI_KL_match.csv")
    parser.add_argument('--val_size', default=1225)
    args = parser.parse_args()

    # Create a dictionary with keys as images names and values as KL scores
    with open(args.matchbook) as infile:
        reader = csv.reader(infile)
        imgdict = {rows[0]: rows[1] for rows in reader}

    # Move all the images in the current validation set to train set
    current_val_list = glob.glob(args.val_path + '/*.png')
    for val_img in current_val_list:
        img_id_tomove = val_img.split("/")[-1]
        shutil.move(args.val_path + '/' + img_id_tomove, args.train_path + '/' + img_id_tomove)

    # Get all image names in the training set, and their corresponding KL scores
    with open(args.matchbook) as infile:
        reader = csv.reader(infile)
        imgdict = {rows[0]: rows[1] for rows in reader}
    train_img = os.listdir(args.train_path)
    train_KL = [imgdict.get(key) for key in train_img]

    # Set seed for reproducibility
    random.seed(42)

    for kl in range(5):
        # Get the frequency and count of images in each KL grade
        num_img_KL = np.sum(np.array(train_KL)==str(kl))
        print("KL=" + str(kl) + ": ", num_img_KL)
        freq_img_KL = num_img_KL / len(train_img)

        # Move "some" images (based on frequency in train set) in current iterating KL grade to validation set
        img_of_this_kl = np.array(train_img)[np.where(np.array(train_KL) == str(kl))]
        random.shuffle(img_of_this_kl)
        for i in range(int(args.val_size * freq_img_KL)):
            img_move_val = img_of_this_kl[i]
            shutil.move(args.train_path + '/' + img_move_val, args.val_path + '/' + img_move_val)
