import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data


class DSpriteMovieDataset(data.Dataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__()

        # image dimensions
        self.width = 32  #  width is x. 0 is left
        self.height = 32  #  height is y. 0 is top
        self.channel = 3
        self.duration = 10

        # latent factors
        # self.shapes = ['triangle', 'square', 'circle']
        self.shapes = ['square']
        self.colours = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        self.initial_sizes = range(2, 8, 2)
        self.max_delta_position = 4
        self.size_growths = [-2, -1, 0, 1, 2]
        self.background_colours = ['black', 'white']
        # list of factors actually varying and modeled in the dataset
        self.latent_factors = {
            'shape': False,
            'colour': True,
            'background_colour': True,
            'size': True,
            'size_growth': False,
        }

        #  mapping symbolic to value
        self.colour_map = {
            'red': torch.tensor([255, 0, 0]).long(),
            'green': torch.tensor([0, 255, 0]).long(),
            'blue': torch.tensor([0, 0, 255]).long(),
            'yellow': torch.tensor([255, 255, 0]).long(),
            'magenta': torch.tensor([255, 0, 255]).long(),
            'cyan': torch.tensor([0, 255, 255]).long(),
            'black': torch.tensor([0, 0, 0]).long(),
            'white': torch.tensor([255, 255, 255]).long()
        }
        return

    def __repr__(self):
        name = f'DSpriteMovie'
        return name

    def __str__(self):
        return f'DSpriteMovie'

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

    def __getitem__(self):
        """
        Generates one sample of data
        """
        # movie
        movie = torch.zeros((self.height, self.width, self.channel, self.duration)).long()

        # randomly draw trajectories for latent factors
        if self.latent_factors['colour']:
            colour = self.colour_map[random.sample(self.colours, 1)[0]]
        else:
            colour = self.colour_map['red']

        if self.latent_factors['background_colour']:
            background_colour = self.colour_map[random.sample(self.background_colours, 1)[0]]
        else:
            background_colour = self.colour_map['white']

        if self.latent_factors['shape']:
            shape = random.sample(self.shapes, 1)
        else:
            shape = 'square'

        if self.latent_factors['size']:
            initial_size = random.sample(self.initial_sizes, 1)[0]
        else:
            initial_size = 4

        if self.latent_factors['size_growth']:
            size_growth = random.sample(self.size_growths)
        else:
            size_growth = 0

        # init position
        initial_delta_x = random.sample(range(-self.max_delta_position, self.max_delta_position), 1)[0]
        initial_delta_y = random.sample(range(-self.max_delta_position, self.max_delta_position), 1)[0]
        initial_direction = (initial_delta_x, initial_delta_y)

        # draw background
        movie[:, :, 0, :] = background_colour[0]
        movie[:, :, 1, :] = background_colour[1]
        movie[:, :, 2, :] = background_colour[2]

        # initial position
        x_init = random.sample(range(initial_size // 2, self.width - initial_size // 2), 1)[0]
        y_init = random.sample(range(initial_size // 2, self.height - initial_size // 2), 1)[0]

        x_t = x_init
        y_t = y_init
        size_t = initial_size
        half_size_t = size_t // 2
        direction_t = initial_direction

        for t in range(self.duration):
            ########################
            #  Draw frame
            if shape == 'square':
                movie[x_t - half_size_t:x_t + half_size_t, y_t - half_size_t:y_t + half_size_t, 0, t] = colour[0]
                movie[x_t - half_size_t:x_t + half_size_t, y_t - half_size_t:y_t + half_size_t, 1, t] = colour[1]
                movie[x_t - half_size_t:x_t + half_size_t, y_t - half_size_t:y_t + half_size_t, 2, t] = colour[2]

            ########################
            #  Get new direction at t+1
            left_tp1 = x_t + direction_t[0] - half_size_t
            right_tp1 = x_t + direction_t[0] + half_size_t
            up_tp1 = y_t + direction_t[1] - half_size_t
            down_tp1 = y_t + direction_t[1] + half_size_t
            if left_tp1 <= 0:
                direction_x = -direction_t[0]
                x_tp1 = -left_tp1 + half_size_t
            elif right_tp1 >= self.width:
                direction_x = -direction_t[0]
                x_tp1 = self.width - (right_tp1 - self.width) - half_size_t
            else:
                direction_x = direction_t[0]
                x_tp1 = x_t + direction_x

            if up_tp1 <= 0:
                direction_y = -direction_t[1]
                y_tp1 = - up_tp1 + half_size_t
            elif down_tp1 >= self.height:
                direction_y = -direction_t[1]
                y_tp1 = self.height - (down_tp1 - self.height) - half_size_t
            else:
                direction_y = direction_t[1]
                y_tp1 = y_t + direction_y

            ########################
            # Compute parameters at t+1
            size_t = size_t
            half_size_t = size_t // 2
            x_t = x_tp1
            y_t = y_tp1
            direction_t = (direction_x, direction_y)
        return movie

    def visualise_batch(self, movies, writing_dir):
        batch_dim = len(movies)

        if os.path.isdir(writing_dir):
            shutil.rmtree(writing_dir)
        os.makedirs(writing_dir)

        for batch_ind in range(batch_dim):
            for time in range(self.duration):
                filepath = f'{writing_dir}/{batch_ind}_{time}.png'
                movie_t = movies[batch_ind, :, :, :, time].numpy()
                plt.imshow(movie_t)
                plt.savefig(filepath)


if __name__ == '__main__':
    dataset = DSpriteMovieDataset()
    movies = []
    for batch_ind in range(100):
        movies.append(dataset.__getitem__())
    movies = torch.stack(movies)
    dataset.visualise_batch(movies, '/home/leo/Recherche/Code/DatasetManager/DatasetManager/dump/dSprite_movie')
