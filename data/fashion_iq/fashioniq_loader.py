from PIL import Image
from os import mkdir, path, listdir
from urllib.request import urlretrieve
import re

def get_images():
    # folder path
    cur_path = path.dirname(path.abspath(__file__))
    dir_path = path.join(cur_path, 'fashion-iq-metadata\image_url')
    output_path = path.join(cur_path, 'images')

    # save all files on the folder `images`
    if not path.exists(output_path):
        mkdir(output_path)

    # get all broken links first
    print("Get all broken links images")
    for file in listdir(path.join(dir_path,"broken_links")):
        if file.endswith('.jpg'):
            img = Image.open(path.join(dir_path, "broken_links", file))
            img.save(path.join(output_path, file))

    # download images
    for file in listdir(dir_path):
        if file.endswith('.txt'):
            f = open(path.join(dir_path, file), "r")
            print(f"Get images from {file}")
            for line in f:
                tmp = re.split(' |\t|\n', line)
                label, link = tmp[0], tmp[-2]
                urlretrieve(link, path.join(output_path, label + '.jpg'))