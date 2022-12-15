from style.style_transfer import create_style_augmented_images
import os
import pandas as pd
from datetime import datetime
import shutil
import argparse


parser = argparse.ArgumentParser(description="PyTorch Style Augmentation")
parser.add_argument(
    "--datadir",
    type=str,
    required=True,
    help="Directory for new datasets. Must contain the raw thermal dataset.",
)
parser.add_argument(
    "--content_weight",
    type=int,
    default=1,
    help="Weight on the content loss. Controls the degree of style transfer.",
)
args = parser.parse_args()


def get_hour(name):
    if "clip" in name:
        time = name.split("_")[3][:2]
    else:
        time = name[8:10]
    return time


def get_minute(name):
    if "clip" in name:
        time = name.split("_")[3][2:4]
    else:
        time = name[10:12]
    return time


# Metadata on images
files = os.listdir(os.path.join(args.datadir, "thermal", "images"))
names = list(set([x.split(".")[0] for x in files]))
names = sorted(names)
filenames = pd.DataFrame(names, columns=["image_name"])
filenames["year"] = filenames.image_name.apply(lambda x: x[:4])
filenames["month"] = filenames.image_name.apply(lambda x: x[4:6])
filenames["day"] = filenames.image_name.apply(lambda x: x[6:8])
filenames["hour"] = filenames.image_name.apply(get_hour)
filenames["minute"] = filenames.image_name.apply(get_minute)
filenames["date"] = (
    "2020"
    + "-"
    + filenames["month"]
    + "-"
    + filenames["day"]
    + "-"
    + filenames["hour"]
    + "-"
    + filenames["minute"]
)
filenames = filenames.sort_values(by=["month", "day", "hour", "minute"])
filenames["dup_count"] = filenames.groupby(["date"]).cumcount() + 1
filenames["date_dup"] = (
    filenames["date"] + "-" + filenames["dup_count"].apply(lambda x: f"{x:02d}")
)


# Rename every image to its datetime
os.mkdir(os.path.join(args.datadir, "renamed"))
os.mkdir(os.path.join(args.datadir, "renamed", "images"))
os.mkdir(os.path.join(args.datadir, "renamed", "labels"))
for i, row in filenames.iterrows():
    old_image = row["image_name"] + ".jpg"
    new_image = row["date_dup"].replace("-", "") + ".jpg"
    old_fp = os.path.join(args.datadir, "thermal/images/", old_image)
    new_fp = os.path.join(args.datadir, "renamed/images/", new_image)
    shutil.copy(old_fp, new_fp)
    old_label = row["image_name"] + ".txt"
    new_label = row["date_dup"].replace("-", "") + ".txt"
    old_fp = os.path.join(args.datadir, "thermal/labels/", old_label)
    new_fp = os.path.join(args.datadir, "renamed/labels/", new_label)
    shutil.copy(old_fp, new_fp)


# Set the months to train a model on
image_names = os.listdir(os.path.join(args.datadir, "renamed", "images"))
dataset_dir = {}
dataset_dir["march"] = {
    "content_start_date": "202001000000",
    "content_end_date": "202002190000",
    "style_start_date": "202002190000",
    "style_end_date": "202003000000",
    "test_dates": ("202003000000", "202004000000"),
}
dataset_dir["april"] = {
    "content_start_date": "202001000000",
    "content_end_date": "202003050000",
    "style_start_date": "202003050000",
    "style_end_date": "202004000000",
    "test_dates": ("202004000000", "202005000000"),
}
dataset_dir["august"] = {
    "content_start_date": "202001000000",
    "content_end_date": "202004200000",
    "style_start_date": "202004200000",
    "style_end_date": "202008000000",
    "test_dates": ("202008000000", "202009000000"),
}
dataset_dir["january"] = {
    "content_start_date": "202008000000",
    "content_end_date": "202009000000",
    "style_start_date": "202001000000",
    "style_end_date": "202001110000",
    "test_dates": ("202001110000", "202002000000"),
}


def candidate_style_imgs(filenames, start_date, end_date):
    """Returns a dataframe with metadata for all images in given time period"""
    candidate = (filenames["date"].str.replace("-", "") >= start_date) & (
        filenames["date"].str.replace("-", "") < end_date
    )
    candidate_df = filenames[candidate]
    candidate_df["hourminute"] = candidate_df["date"].apply(
        lambda x: x.replace("-", "")[8:12]
    )
    candidate_df["timeofday"] = candidate_df["hourminute"].apply(
        lambda x: datetime.strptime(x, "%H%M")
    )
    return candidate_df


def assign_style_img(content_img, candidate_df):
    """Find the style image that is nearest in time of day"""
    content_time = datetime.strptime(content_img[8:12], "%H%M")
    candidate_df["difference"] = candidate_df["timeofday"] - content_time
    candidate_df["difference"] = candidate_df["difference"].abs()
    argmin = candidate_df["difference"].argmin()
    style_img = candidate_df.iloc[argmin]["date_dup"].replace("-", "")
    return style_img


# Assign a style image to each content images
for month in ["march", "april", "august", "january"]:
    dataset_dir[month]["style_assignments"] = {}
    candidate_df = candidate_style_imgs(
        filenames,
        start_date=dataset_dir[month]["style_start_date"],
        end_date=dataset_dir[month]["style_end_date"],
    )
    for content_img in image_names:
        content_img = content_img.split(".")[0]
        if (content_img < dataset_dir[month]["content_start_date"]) or (
            content_img > dataset_dir[month]["content_end_date"]
        ):
            continue
        style_img = assign_style_img(content_img, candidate_df)
        if style_img not in dataset_dir[month]["style_assignments"]:
            dataset_dir[month]["style_assignments"][style_img] = []
        dataset_dir[month]["style_assignments"][style_img].append(content_img)


for month in ["march", "april", "august", "january"]:
    os.path.join(args.datadir, month)
    os.mkdir(os.path.join(args.datadir, month))
    os.mkdir(os.path.join(args.datadir, month, "train"))
    os.mkdir(os.path.join(args.datadir, month, "test"))
    os.mkdir(os.path.join(args.datadir, month, "train", "images"))
    os.mkdir(os.path.join(args.datadir, month, "train", "labels"))
    os.mkdir(os.path.join(args.datadir, month, "test", "images"))
    os.mkdir(os.path.join(args.datadir, month, "test", "labels"))
    style_start_date = dataset_dir[month]["style_start_date"]
    style_end_date = dataset_dir[month]["style_end_date"]
    test_start, test_end = dataset_dir[month]["test_dates"]
    # Add stylized images to train set
    for style_img_name, content_img_names in dataset_dir[month][
        "style_assignments"
    ].items():
        create_style_augmented_images(
            style_img_name=style_img_name,
            content_img_names=content_img_names,
            image_dir=os.path.join(args.datadir, "renamed", "images"),
            output_dir=os.path.join(args.datadir, month, "train", "images"),
            num_steps=300,
            content_weight=args.content_weight,
        )
    # Add in unstylized images to train set
    unstylized_img_names = [
        x.split(".")[0]
        for x in image_names
        if (x.split(".")[0] >= style_start_date) and (x.split(".")[0] < style_end_date)
    ]
    for img in unstylized_img_names:
        old_fp = os.path.join(args.datadir, "renamed", "images", img + ".jpg")
        new_fp = os.path.join(args.datadir, month, "train", "images", img + ".jpg")
        shutil.copy(old_fp, new_fp)
    # Add images to test set
    test_img_names = [
        x.split(".")[0]
        for x in image_names
        if (x.split(".")[0] >= test_start) and (x.split(".")[0] < test_end)
    ]
    for img in test_img_names:
        old_fp = os.path.join(args.datadir, "renamed", "images", img + ".jpg")
        new_fp = os.path.join(args.datadir, month, "test", "images", img + ".jpg")
        shutil.copy(old_fp, new_fp)

# Add labels
for month in ["march", "april", "august", "january"]:
    for split in ["train", "test"]:
        img_names = os.listdir(os.path.join(args.datadir, month, split, "images"))
        img_names = [x for x in img_names if x[0] != "."]
        for img in img_names:
            name = img.split(".")[0]
            old_fp = os.path.join(args.datadir, "renamed", "labels", name + ".txt")
            new_fp = os.path.join(args.datadir, month, split, "labels", name + ".txt")
            shutil.copy(old_fp, new_fp)

# Create baseline datasets
image_names = os.listdir(os.path.join(args.datadir, "renamed", "images"))
image_names = [x for x in image_names if x[0] != "."]
for month in ["march", "april", "august", "january"]:
    os.mkdir(os.path.join(args.datadir, month + "_unstylized"))
    os.mkdir(os.path.join(args.datadir, month + "_unstylized", "train"))
    os.mkdir(os.path.join(args.datadir, month + "_unstylized", "test"))
    os.mkdir(os.path.join(args.datadir, month + "_unstylized", "train", "images"))
    os.mkdir(os.path.join(args.datadir, month + "_unstylized", "train", "labels"))
    os.mkdir(os.path.join(args.datadir, month + "_unstylized", "test", "images"))
    os.mkdir(os.path.join(args.datadir, month + "_unstylized", "test", "labels"))
    # Train dataset
    train_names = os.listdir(os.path.join(args.datadir, month, "train", "images"))
    train_names = [x.split(".")[0] for x in train_names if x[0] != "."]
    for name in train_names:
        old_fp = os.path.join(args.datadir, "renamed", "images", name + ".jpg")
        new_fp = os.path.join(
            args.datadir, month + "_unstylized", "train", "images", name + ".jpg"
        )
        shutil.copy(old_fp, new_fp)
        old_fp = os.path.join(args.datadir, "renamed", "labels", name + ".txt")
        new_fp = os.path.join(
            args.datadir, month + "_unstylized", "train", "labels", name + ".txt"
        )
        shutil.copy(old_fp, new_fp)
    test_names = os.listdir(os.path.join(args.datadir, month, "test", "images"))
    test_names = [x.split(".")[0] for x in test_names if x[0] != "."]
    for name in test_names:
        old_fp = os.path.join(args.datadir, "renamed", "images", name + ".jpg")
        new_fp = os.path.join(
            args.datadir, month + "_unstylized", "test", "images", name + ".jpg"
        )
        shutil.copy(old_fp, new_fp)
        old_fp = os.path.join(args.datadir, "renamed", "labels", name + ".txt")
        new_fp = os.path.join(
            args.datadir, month + "_unstylized", "test", "labels", name + ".txt"
        )
        shutil.copy(old_fp, new_fp)
