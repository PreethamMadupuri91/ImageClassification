import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras import layers, Model

class Solution(object):

    def main(self):

        base_dir = '/Users/preethamkumarmadupuri/Desktop/Computer_Vision/cats_and_dogs_filtered/cats_and_dogs_filtered'
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

        train_cats_dir = os.path.join(train_dir, 'cats')
        train_dogs_dir = os.path.join(train_dir, 'dogs')

        validation_cats_dir = os.path.join(validation_dir, 'cats')
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')

        train_cat_fnames = os.listdir(train_cats_dir)
        print(train_cat_fnames[:10])

        train_dog_fnames = os.listdir(train_dogs_dir)
        train_dog_fnames.sort()
        print(train_dog_fnames[:10])

        print('total training cat images:', len(os.listdir(train_cats_dir)))
        print('total training dog images:', len(os.listdir(train_dogs_dir)))
        print('total validation cat images:', len(os.listdir(validation_cats_dir)))
        print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

        nrows = 4
        ncols = 4

        pic_index = 0

        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)

        pic_index += 8

        next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index-8:pic_index]]
        next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index-8:pic_index]]

        for i, img_path in enumerate(next_cat_pix+next_dog_pix):
        # Set up subplot; subplot indices start at 1
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

        plt.show()

        img_input = layers.Input(shape=(150, 150, 3))

        # First convolution extracts 16 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(16, 3, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)

        # Second convolution extracts 32 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Third convolution extracts 64 filters that are 3x3
        # Convolution is followed by max-pooling layer with a 2x2 window
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

        # Flatten feature map to a 1-dim tensor so we can add fully connected layers
        x = layers.Flatten()(x)

        # Create a fully connected layer with ReLU activation and 512 hidden units
        x = layers.Dense(512, activation='relu')(x)

        # Create output layer with a single node and sigmoid activation
        output = layers.Dense(1, activation='sigmoid')(x)

        # Create model:
        # input = input feature map
        # output = input feature map + stacked convolution/maxpooling layers + fully
        # connected layer + sigmoid output layer
        model = Model(img_input, output)
        model.summary()

    if __name__ == "__main__":
       main(object)