#%%

from style_transfer import create_style_augmented_images
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import shutil


files = os.listdir('datasets/thermal/images')
names = list(set([x.split('.')[0] for x in files]))
names = sorted(names)

def get_hour(name):
    if 'clip' in name:
        time = name.split('_')[3][:2]
    else:
        time = name[8:10]
    return time

def get_minute(name):
    if 'clip' in name:
        time = name.split('_')[3][2:4]
    else:
        time = name[10:12]
    return time
    
filenames = pd.DataFrame(names, columns=['image_name'])
filenames['year'] = filenames.image_name.apply(lambda x: x[:4])
filenames['month'] = filenames.image_name.apply(lambda x: x[4:6])
filenames['day'] = filenames.image_name.apply(lambda x: x[6:8])
filenames['hour'] = filenames.image_name.apply(get_hour)
filenames['minute'] = filenames.image_name.apply(get_minute)
filenames['date'] = "2020"+'-'+filenames['month']+'-'+filenames['day']+'-'+filenames['hour']+'-'+filenames['minute']
filenames = filenames.sort_values(by=['month', 'day', 'hour', 'minute'])

filenames = filenames[(filenames.hour >= '07') & (filenames.hour <= '19')] 


#%%

dates = [datetime.strptime(d, "%Y-%m-%d-%H-%M") for d in filenames['date'].to_numpy()]

fig, ax = plt.subplots()
y = np.random.uniform(0,1,len(dates)) #np.zeros_like(dates)
y = np.random.randn(len(dates))
y = np.clip(y, -3,3)
ax.scatter(dates, y, s=5)
ax.set_ylim(-4,5)
ax.set_yticks([])
plt.show()


# Evaluate on: August (08), April (04), March (03)
# apply style to all images 15 days before the cutoff date

# %%

os.mkdir('datasets/thermal/renamed/')
for i, row in filenames.iterrows():
    print(i, row)
    image_name = row['image_name']+'.jpg'
    new_name = row['date'].replace('-','')+'.jpg'
    new_fp = os.path.join('datasets/thermal/renamed/', new_name)
    old_fp = os.path.join('datasets/thermal/images/', image_name)
    shutil.copy(old_fp, new_fp)


# %%

image_names = os.listdir('datasets/thermal/renamed')
dataset_dir = {}

dataset_dir['march'] = {}
dataset_dir['march']['style_image_names'] = ['202002271324', '202002270946']
dataset_dir['march']['content_image_names'] = [x for x in image_names if x.split('.')[0] < '202002190000']

dataset_dir['april'] = {}
dataset_dir['april']['style_image_names'] = ['202003101350', '202003091658']
dataset_dir['april']['content_image_names'] = [x for x in image_names if x.split('.')[0] < '202003050000']

dataset_dir['august'] = {}
dataset_dir['august']['style_image_names'] = ['202004241610', '202004231257']
dataset_dir['august']['content_image_names'] = [x for x in image_names if x.split('.')[0] < '202004200000']


for month in ['march', 'april', 'august']:
    for style_img_name in dataset_dir[month]['style_image_names']:
        content_img_names = dataset_dir[month]['content_image_names']
        create_style_augmented_images(
            style_img_name=style_img_name,
            content_img_names=content_img_names,
            image_dir="datasets/thermal/renamed",
            output_dir="datasets/stylized",
            num_steps=300,
        )
