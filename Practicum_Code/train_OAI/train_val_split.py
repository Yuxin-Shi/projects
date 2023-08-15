import os, random
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainSet', default="C:/University/2022_Fall/CHL5207/Knee_Osteoarthritis"
                                              "/OAI_raw/OAI_cropped_auto/train")
    parser.add_argument('--valSet', default="C:/University/2022_Fall/CHL5207/Knee_Osteoarthritis"
                                              "/OAI_raw/OAI_cropped_auto/val")
    parser.add_argument('--valSize', default=1500)
    args = parser.parse_args()

    train_img_name = random.sample(os.listdir(args.trainSet), args.valSize)
    print(len(train_img_name))

    for img in train_img_name:
        train_img_full_path = os.path.join(args.trainSet, img)
        val_img_full_path = os.path.join(args.valSet, img)
        shutil.move(train_img_full_path, val_img_full_path)




