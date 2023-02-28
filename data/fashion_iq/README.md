To start, first install the requirements from the requirements.txt using the following command:
`pip install -r data\fashion-iq\requirements.txt`

After that, you can simply run `fashion-iq-loader.py` to get the Fashion IQ image dataset into the folder `images`. `captions` and `image_splits` are available in json format in the folder `fashion-iq`. Usually models would utilize three of these folders for training.

Finally, you can now run `train_fashioniq.py` in the main folder.

- fashion-iq is cloned from this [git repo](https://github.com/XiaoxiaoGuo/fashion-iq.git)
- fashion-iq-metadata is cloned from this [git repo](https://github.com/hongwang600/fashion-iq-metadata.git)

