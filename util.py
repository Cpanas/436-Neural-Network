import numpy as np # type: ignore
import os
from PIL import Image as img # type: ignore
import matplotlib.pyplot as plt # type: ignore
import main
import time
import random

pix = 0

def data_request(dir):
    dir_sub = os.listdir(dir)
    count = 0

    for sub in dir_sub:
        for file in os.listdir(dir + "/" + sub):
            count = count + 1

    return len(dir_sub), count

def build_data(dir, samples, split, train):
    print("Building Data [Dir: {} Samples: {} Split: {} Is_training: {}]".format(dir, samples, split, train))

    global pix
    
    x_data_table = []
    y_data_table = []

    total = 0
    iteration = 0

    #define the label hash
    if len(main.label_hash) == 0:
        main.label_hash = dict(zip(os.listdir(dir), range(0,len(os.listdir(dir)))))

    #Label is sub dir
    for sub in os.listdir(dir):
        #trim the amount of samples we want
        subdir = os.listdir(dir + "/" + sub)

        #split the data between training and test
        if train == True:
            subdir = subdir[0:int((len(subdir) - 1) * split)]
        else:
            subdir = subdir[int((len(subdir) - 1) * split): len(subdir) - 1]

        #randomize what we pull from this pool
        random.shuffle(subdir)

        #pull only the amount of samples we want from the pool
        if samples > 0 and len(subdir) > samples:
            subdir = subdir[0:samples]

            #total is used for the loading bar - set this here
            total = len(os.listdir(dir)) * samples
        else:
            total = len(os.listdir(dir)) * len(subdir)

        for file in subdir:
            if pix == 0:
                pix = np.array(img.open(dir + "/" + sub + "/" + file).convert('L')).shape
            x_data_table.append(np.array(img.open(dir + "/" + sub + "/" + file).convert('L')).flatten()/255)
            #TODO create hashmap for directory names, so the directories don't need to be numbers
            y_data_table.append(main.label_hash[sub])

            if main.progress_bar is True:
                iteration = iteration + 1
                progress_bar(iteration, total)

    return np.array(x_data_table), np.array(y_data_table)

#Not mine! IDR where I got this, helpful for shuffling the np arrays
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#Images come out as purple for some reason
#TODO make images not purple
def rebuild_image(x_data, sleep = -1):
    global pix
    size = pix
    x_data = x_data * 255
    reformated = x_data.reshape(size)
    image = img.new('L', (size))
    pic = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pic[x, y] = int(reformated[y][x])

    plt.imshow(image)

    if sleep > 0:
        plt.pause(sleep)

def get_predictions(X):
    return np.argmax(X, 0)

def get_accuracy(pred, Y):
    return np.sum(pred == Y) / Y.size

# Print progress - i did not make this!
#Credit https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progress_bar(iteration, total):

    percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(100 * iteration // total)
    bar = '█' * filledLength + '-' * (100 - filledLength)
    print(f'\r|{bar}| {percent}%', end = "\r")
    # Print New Line on Complete
    if iteration == total: 
        print()

def row_average(arr):
    ret = []
    for column in arr.T:
        ret.append(sum(column))
    return ret

    
