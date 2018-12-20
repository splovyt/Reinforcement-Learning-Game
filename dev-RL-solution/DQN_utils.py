'''This file contains the utility functions used in the DQN reinforcement learning agent.'''

from PIL import Image
import numpy as np


def rgb2gray(rgb):
    '''Convert an RGB-array to grayscale.'''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def reshape_state(img, new_height, new_width):
    '''Reshape the state to have the correct input properties for the neural network.'''
    # resize image
    resized_img = np.array(Image.fromarray(img).resize((new_width, new_height)))
    # make grayscale
    resized_gray = rgb2gray(resized_img)
    # normalize
    resized_gray_normalized = resized_gray / 255.0
    # reshape for the network
    reshaped = resized_gray_normalized.reshape(1, new_height, new_width, 1)
    return reshaped