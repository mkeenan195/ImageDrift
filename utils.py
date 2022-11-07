import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

DATA_ROOT = 'C:/Users/mkeen/Desktop/CSCI-GA_3033_Intro_DL_Systems/ImageDrift/data/'


def get_bounding_boxes(image_name):
    if image_name.split('.')[-1]=='jpg':
        image_name = image_name[:-4]
    try:
        df = pd.read_csv(f'{DATA_ROOT}/{image_name}.txt', sep=' ', header=None)
        df.columns =['i','x','y','w','h']
    except pd.errors.EmptyDataError:
        return None
    return df
    

def data_coords_to_plt_coords(x,y,w,h):
    xc = 384
    yc = 288
    w = round(w*xc)
    h = round(h*yc)
    x = round(x*xc-0.5*w)
    y = round(y*yc-0.5*h)    
    return x, y, w, h


def plot_image(image_name, title=None):
    if image_name[-3:] != 'jpg':
        image_name = image_name+'.jpg'
    img = mpimg.imread(f'{DATA_ROOT}/{image_name}')
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    bounding_boxes = get_bounding_boxes(image_name)
    if bounding_boxes is not None:
        for i, row in bounding_boxes.iterrows():
            x = row.x
            y = row.y
            w = row.w
            h = row.h 
            x, y, w, h = data_coords_to_plt_coords(x, y, w, h)
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.show()


if __name__=='__main__':
    files = os.listdir(DATA_ROOT)
    names = list(set([x.split('.')[0] for x in files]))
    names = sorted(names)
    
    filenames = pd.DataFrame(names, columns=['image_name'])
    filenames['month'] = filenames.image_name.apply(lambda x: x[4:6])
    filenames['year'] = filenames.image_name.apply(lambda x: x[:4])
    filenames = filenames.sort_values(by=['month'])
    
    
    for image_name in filenames.image_name:
        print(image_name)
        if 'clip' in image_name:
            date = image_name.split('_')[0]
            time = image_name.split('_')[3]
        else:
            date = image_name[:8]
            time = image_name[8:12]
        if time>='01000' and time<='1400':
            plot_image(image_name, title=f'{date}, {time}')
    
    
    
