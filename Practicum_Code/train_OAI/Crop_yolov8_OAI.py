from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageOps
import glob
import argparse
import random
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/project/o/oespinga/shiyuxin/OAI_images/best.pt")
    parser.add_argument('--img_folder', default="/project/o/oespinga/shiyuxin/OAI_images/1205728_P001")
    parser.add_argument('--train_path', default="/project/o/oespinga/shiyuxin/OAI_images/1205728_P001_yolov8/train")
    parser.add_argument('--val_path', default="/project/o/oespinga/shiyuxin/OAI_images/1205728_P001_yolov8/val")
    parser.add_argument('--val_size', default=1500)
    parser.add_argument('--output_size', default=220)
    args = parser.parse_args()

    # Load the pre-trained model
    model = YOLO(args.model_path)
    # Load raw image list
    jpg_list = glob.glob(args.img_folder + "/*.jpg")
    random.shuffle(jpg_list)

    # Crop each image in image list
    crop_error = []
    for img_path in jpg_list:
        img = Image.open(img_path)

        # Predict on OAI images
        result = model.predict(img_path)
        if len(result[0].boxes) < 2:
            crop_error.append(img_path.split("/")[-1])
            continue

        box1 = result[0].boxes[0]
        cords1 = box1.xyxy[0].tolist()
        box2 = result[0].boxes[1]
        cords2 = box2.xyxy[0].tolist()

        # Make cropped image to have size output_size * output_size
        # center1 = ((cords1[0] + cords1[2]) / 2, (cords1[1] + cords1[3]) / 2)
        # center2 = ((cords2[0] + cords2[2]) / 2, (cords2[1] + cords2[3]) / 2)
        #
        # adj_cords1 = (center1[0] - args.output_size / 2, center1[1] - args.output_size / 2,
        #               center1[0] + args.output_size / 2, center1[1] + args.output_size / 2)
        # adj_cords2 = (center2[0] - args.output_size / 2, center2[1] - args.output_size / 2,
        #               center2[0] + args.output_size / 2, center2[1] + args.output_size / 2)
        #
        # crop1 = img.crop(adj_cords1)
        # crop2 = img.crop(adj_cords2)

        crop1 = img.crop(cords1)
        crop1 = crop1.resize((args.output_size, args.output_size))
        crop2 = img.crop(cords2)
        crop2 = crop2.resize((args.output_size, args.output_size))

        if crop1.mode == "F":
            crop1 = crop1.convert("RGB")
        if crop2.mode == "F":
            crop2 = crop2.convert("RGB")

        # Save all cropped images (knee joints) to train set
        image_id = img_path.split("/")[-1].split(".")[0]
        if cords1[0] < cords2[0]:
            crop1.save(args.train_path + "/" + image_id + "_R.png")
            crop2.save(args.train_path + "/" + image_id + "_L.png")
        else:
            crop1.save(args.train_path + "/" + image_id + "_L.png")
            crop2.save(args.train_path + "/" + image_id + "_R.png")

    # Move some images from train set to validation set
    all_img_list = glob.glob(args.train_path + '/*.png')
    for i in range(args.val_size):
        img_id_tomove = all_img_list[i].split("/")[-1]
        shutil.move(args.train_path + '/' + img_id_tomove, args.val_path + '/' + img_id_tomove)

    # Print all the error cropped image names
    print(*crop_error, sep="\n")
