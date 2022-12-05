from style_transfer import create_style_augmented_images
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import shutil

DATASET_DIR = '/scratch_tmp/mpk9358/datasets'

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
files = os.listdir(os.path.join(DATASET_DIR, 'thermal','images'))
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



# Plot distribution of image dates
# dates = [datetime.strptime(d, "%Y-%m-%d-%H-%M") for d in filenames['date'].to_numpy()]
# fig, ax = plt.subplots()
# y = np.random.uniform(0,1,len(dates)) #np.zeros_like(dates)
# y = np.random.randn(len(dates))
# y = np.clip(y, -3,3)
# ax.scatter(dates, y, s=5)
# ax.set_ylim(-4,5)
# ax.set_yticks([])
# plt.show()


# Rename every image to its datetime
# os.mkdir(os.path.join(DATASET_DIR, 'renamed'))
# os.mkdir(os.path.join(DATASET_DIR, 'renamed','images'))
# os.mkdir(os.path.join(DATASET_DIR, 'renamed','labels'))
# for i, row in filenames.iterrows():
#     old_image = row["image_name"] + ".jpg"
#     new_image = row["date_dup"].replace("-", "") + ".jpg"
#     old_fp = os.path.join(DATASET_DIR, "thermal/images/", old_image)
#     new_fp = os.path.join(DATASET_DIR, "renamed/images/", new_image)
#     shutil.copy(old_fp, new_fp)
#     old_label = row["image_name"] + ".txt"
#     new_label = row["date_dup"].replace("-", "") + ".txt"
#     old_fp = os.path.join(DATASET_DIR, "thermal/labels/", old_label)
#     new_fp = os.path.join(DATASET_DIR, "renamed/labels/", new_label)
#     shutil.copy(old_fp, new_fp)



# Set the months to train a model on
image_names = os.listdir(os.path.join(DATASET_DIR, 'renamed', 'images'))
dataset_dir = {}
dataset_dir["march"] = {
    "content_end_date": "202002190000",
    "dataset_end_date": "202003000000",
    "test_dates": ("202003000000", "202004000000")
}
dataset_dir["april"] = {
    "content_end_date": "202003050000",
    "dataset_end_date": "202004000000",
    "test_dates": ("202004000000", "202005000000")
}
dataset_dir["august"] = {
    "content_end_date": "202004200000",
    "dataset_end_date": "202008000000",
    "test_dates": ("202008000000", "202009000000")
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
for month in ["march", "april", "august"]:
    dataset_dir[month]["style_assignments"] = {}
    candidate_df = candidate_style_imgs(
        filenames,
        start_date=dataset_dir[month]["content_end_date"],
        end_date=dataset_dir[month]["dataset_end_date"],
    )
    for content_img in image_names:
        content_img = content_img.split('.')[0]
        if content_img > dataset_dir[month]["content_end_date"]:
            continue
        style_img = assign_style_img(content_img, candidate_df)
        if style_img not in dataset_dir[month]["style_assignments"]:
            dataset_dir[month]["style_assignments"][style_img] = []
        dataset_dir[month]["style_assignments"][style_img].append(content_img)


for month in ["august"]: #["march", "april", "august"]:
    os.path.join(DATASET_DIR, month)
    os.mkdir(os.path.join(DATASET_DIR, month))
    os.mkdir(os.path.join(DATASET_DIR, month, "train"))
    os.mkdir(os.path.join(DATASET_DIR, month, "test"))
    os.mkdir(os.path.join(DATASET_DIR, month, "train", "images"))
    os.mkdir(os.path.join(DATASET_DIR, month, "train", "labels"))
    os.mkdir(os.path.join(DATASET_DIR, month, "test", "images"))
    os.mkdir(os.path.join(DATASET_DIR, month, "test", "labels"))
    dataset_end_date = dataset_dir[month]["dataset_end_date"]
    content_end_date = dataset_dir[month]["content_end_date"]
    test_start, test_end = dataset_dir[month]["test_dates"]
    # Add stylized images to train set
    for style_img_name, content_img_names in dataset_dir[month]["style_assignments"].items():
        create_style_augmented_images(
            style_img_name=style_img_name,
            content_img_names=content_img_names,
            image_dir=os.path.join(DATASET_DIR, 'renamed', "images"),
            output_dir=os.path.join(DATASET_DIR, month, "train", "images"),
            num_steps=300,
            content_weight=10
        )
    # Add in unstylized images to train set
    unstylized_img_names = [
        x.split(".")[0]
        for x in image_names
        if (x.split(".")[0] >= content_end_date)
        and (x.split(".")[0] < dataset_end_date)
    ]
    for img in unstylized_img_names:
        old_fp = os.path.join(DATASET_DIR, "renamed", "images", img+'.jpg')
        new_fp = os.path.join(DATASET_DIR, month, "train", "images", img+'.jpg')
        shutil.copy(old_fp, new_fp)
    # Add images to test set
    test_img_names = [
        x.split(".")[0]
        for x in image_names
        if (x.split(".")[0] >= test_start)
        and (x.split(".")[0] < test_end)
    ]
    for img in test_img_names:
        old_fp = os.path.join(DATASET_DIR, "renamed", "images", img+'.jpg')
        new_fp = os.path.join(DATASET_DIR, month, "test", "images", img+'.jpg')
        shutil.copy(old_fp, new_fp)

# Add labels
for month in ["march", "april", "august"]:
    for split in ['train','test']:
        img_names = os.listdir(os.path.join(DATASET_DIR, month, split,'images'))
        for img in img_names:
            name = img.split('.')[0]
            old_fp = os.path.join(DATASET_DIR, "renamed", "labels", name+'.txt')
            new_fp = os.path.join(DATASET_DIR,month,split,'labels', name+'.txt')
            shutil.copy(old_fp, new_fp)        
    
# Create baseline datasets
image_names = os.listdir(os.path.join(DATASET_DIR, 'renamed', 'images'))
for month in ["march", "april", "august"]:
    os.mkdir(os.path.join(DATASET_DIR, month+'_unstylized'))
    os.mkdir(os.path.join(DATASET_DIR, month+'_unstylized', 'train'))
    os.mkdir(os.path.join(DATASET_DIR, month+'_unstylized', 'test'))
    os.mkdir(os.path.join(DATASET_DIR, month+'_unstylized', 'train', 'images'))
    os.mkdir(os.path.join(DATASET_DIR, month+'_unstylized', 'train', 'labels'))
    os.mkdir(os.path.join(DATASET_DIR, month+'_unstylized', 'test', 'images'))
    os.mkdir(os.path.join(DATASET_DIR, month+'_unstylized', 'test', 'labels'))
    dataset_end_date = dataset_dir[month]["dataset_end_date"]
    content_end_date = dataset_dir[month]["content_end_date"]
    test_start, test_end = dataset_dir[month]["test_dates"]
    # Train dataset
    train_names = [x.split('.')[0] for x in image_names if x.split('.')[0] < dataset_end_date]
    for name in train_names:
        old_fp = os.path.join(DATASET_DIR, "renamed", "images", name+'.jpg')
        new_fp = os.path.join(DATASET_DIR,month+'_unstylized','train','images', name+'.jpg')
        shutil.copy(old_fp, new_fp)  
        old_fp = os.path.join(DATASET_DIR, "renamed", "labels", name+'.txt')
        new_fp = os.path.join(DATASET_DIR,month+'_unstylized','train','labels', name+'.txt')
        shutil.copy(old_fp, new_fp)  
    test_names = [x.split('.')[0] for x in image_names if (x.split('.')[0] >= test_start) and (x.split('.')[0] < test_end)]
    for name in test_names:
        old_fp = os.path.join(DATASET_DIR, "renamed", "images", name+'.jpg')
        new_fp = os.path.join(DATASET_DIR,month+'_unstylized','test','images', name+'.jpg')
        shutil.copy(old_fp, new_fp)  
        old_fp = os.path.join(DATASET_DIR, "renamed", "labels", name+'.txt')
        new_fp = os.path.join(DATASET_DIR,month+'_unstylized','test','labels', name+'.txt')
        shutil.copy(old_fp, new_fp)  


