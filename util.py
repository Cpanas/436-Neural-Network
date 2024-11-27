import numpy as np # type: ignore
import os
from PIL import Image as img # type: ignore
import matplotlib.pyplot as plt # type: ignore
import main
import time

fig = plt.figure()
ax = fig.add_subplot()

def data_request(train, test):
    train_sub = os.listdir(train)
    test_sub = os.listdir(test)
    train_count = 0
    test_count = 0

    for sub in train_sub:
        for file in os.listdir(test + "/" + sub):
            test_count = test_count + 1

    for sub in test_sub:
        for file in os.listdir(train + "/" + sub):
            train_count = train_count + 1

    return len(train_sub), train_count, len(test_sub), test_count


def build_data(dir, samples = -1):
    print("Building Data [Dir: {} Samples: {}]".format(dir, samples))
    
    x_data_table = []
    y_data_table = []

    total = 0
    iteration = 0

    #Label is sub dir
    for sub in os.listdir(dir):
        #trim the amount of samples we want
        subdir = os.listdir(dir + "/" + sub)
        if samples > 0 and len(subdir) > samples:
            subdir = subdir[0:samples]
            total = len(os.listdir(dir)) * samples
        else:
            total = len(os.listdir(dir)) * len(subdir)

        for file in subdir:
            x_data_table.append(np.array(img.open(dir + "/" + sub + "/" + file).convert('L')).flatten()/255)
            #TODO create hashmap for directory names, so the directories don't need to be numbers
            y_data_table.append(int(sub))

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
def rebuild_image(x_data, size = 28, sleep = -1):
    x_data = x_data * 255
    reformated = x_data.reshape(size , -1)
    image = img.new('L', (size, size))
    pix = image.load()
    for y in range(size):
        for x in range(size):
            pix[x, y] = int(reformated[y][x])

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

    
