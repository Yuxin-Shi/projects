import os
import argparse
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchbook', default="/project/o/oespinga/shiyuxin/OAI_images/OAI_KL_match.csv")
    parser.add_argument('--train', default='/project/o/oespinga/shiyuxin/OAI_images/1205728_P001_dataset/train')
    parser.add_argument('--val', default='/project/o/oespinga/shiyuxin/OAI_images/1205728_P001_dataset/val')
    parser.add_argument('--remove', default=False)
    args = parser.parse_args()

    with open(args.matchbook) as infile:
        reader = csv.reader(infile)
        imgdict = {rows[0]: rows[1] for rows in reader}

    for img_train in os.listdir(args.train):
        try:
            key = imgdict[img_train]
            if key not in ["0", "1", "2", "3", "4"]:
                print("train image KL is NA: " + img_train)
                if args.remove:
                    os.remove(os.path.join(args.train, img_train))
        except KeyError:
            print("train image KL not found: " + img_train)
            if args.remove:
                os.remove(os.path.join(args.train, img_train))
            pass

    for img_val in os.listdir(args.val):
        try:
            key = imgdict[img_val]
            if key not in ["0", "1", "2", "3", "4"]:
                print("val image KL is NA: " + img_val)
                if args.remove:
                    os.remove(os.path.join(args.val, img_val))
        except KeyError:
            print("val image KL not found: " + img_val)
            if args.remove:
                os.remove(os.path.join(args.val, img_val))
            pass

