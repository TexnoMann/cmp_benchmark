import os, sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.air_hockey_challenge.air_hockey_challenge.framework import AgentBase

class DummyAgent(AgentBase):
    def __init__(self, env_info, value, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        self.primitive_variable = value  # Primitive python variable
        self.numpy_vector = np.array([1, 2, 3]) * value  # Numpy array
        self.list_variable = [1, 'list', [2, 3]]  # Numpy array

        # Dictionary
        self.dictionary = dict(some='random', keywords=2, fill='the dictionary')

        # Building a torch object
        data_array = np.ones(3) * value
        data_tensor = torch.from_numpy(data_array)
        self.torch_object = torch.nn.Parameter(data_tensor)

        # A non serializable object
        self.object_instance = object()

        # A variable that is not important e.g. a buffer
        self.not_important = np.zeros(10000)

        # Here we specify how to save each component
        self._add_save_attr(
            primitive_variable='primitive',
            numpy_vector='numpy',
            list_variable='primitive',
            dictionary='pickle',
            torch_object='torch',
            object_instance='none',
            # The '!' is to specify that we save the variable only if full_save is True
            not_important='numpy!',
        )

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        print(self.get_joint_pos(observation))
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        velocity = np.zeros_like(self.hold_position)
        action = np.vstack([self.hold_position, velocity])
        return action