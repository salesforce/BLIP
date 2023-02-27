from PIL import Image
from os import mkdir, path, listdir
from urllib.request import urlretrieve
import re

# folder path
dir_path = './fashion-iq-metadata/image_url'

# save all files on the folder `images`
if not path.exists("images"):
    mkdir("images")

# get all broken links first
print("Get all broken links images")
for file in listdir(path.join(dir_path,"broken_links")):
    if file.endswith('.jpg'):
        img = Image.open(path.join(dir_path, "broken_links", file))
        img.save(path.join("images", file))

for file in listdir(dir_path):
    if file.endswith('.txt'):
        f = open(path.join(dir_path, file), "r")
        print(f"Get images from {file}")
        for line in f:
            tmp = re.split(' |\t|\n', line)
            label, link = tmp[0], tmp[-2]
            urlretrieve(link, path.join('images', label + '.jpg'))