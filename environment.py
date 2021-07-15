from abc import ABC, abstractmethod
import random

import pandas as pd


# TODO current code relies on hist changing size each iteration, which is bad
# Also, constantly constructing new dataframes is slow


class EnvironmentBase(ABC):
    # Environment classes contain methods to get baseline, input, and sustain FRs as well as reward
    # I'm assuming the history dataframe contains columns for State, Action, and Reward, where State
    # can be anything used by the user to keep track of state
    @abstractmethod
    def get_baseline(self, hist):
        # Takes a history dataframe and returns a dataframe with a column for each action channel
        # History dataframe contains a row for each iteration and columns 'State', 'Action', 'Reward'
        pass

    @abstractmethod
    def get_input(self, hist):
        # Takes a history dataframe and returns a dataframe with a column for each action channel
        # as well as a 'State' column, which can be anything storable in a dataframe
        # History dataframe contains a row for each iteration and columns 'State', 'Action', 'Reward'
        pass

    @abstractmethod
    def get_sustain(self, hist):
        # Takes a history dataframe and returns a dataframe with a column for each action channel
        # History dataframe contains a row for each iteration and columns 'State', 'Action', 'Reward'
        pass

    @abstractmethod
    def get_reward(self, hist):
        # Takes a history dataframe and returns a numerical reward value
        # History dataframe contains a row for each iteration and columns 'State', 'Action', 'Reward'
        pass
        # TODO should this return a df with a reward for each action?

class ConstantEnvironment(EnvironmentBase):
    def __init__(self, baseline, input_channel_0, input_other_channels,
            sustain_selected_channel, sustain_other_channels, 
            correct_reward, incorrect_reward, reward_prob=1, channels=2):
        self.baseline = baseline # Baseline input FR
        self.input_channel_0 = input_channel_0 # FR in channel 0 during input phase
        self.input_other_channels = input_other_channels # FR in other channels during input phase
        self.sustain_selected_channel = sustain_selected_channel # FR in selected channel during sustain phase
        self.sustain_other_channels = sustain_other_channels # FR in other channels during sustain phase
        self.correct_reward = correct_reward # Reward for selecting channel 0
        self.incorrect_reward = incorrect_reward # Reward for selecting other channels
        self.reward_prob = reward_prob # Probability of getting the right reward
        self.channels = channels # Number of channels

    def get_baseline(self, hist):
        return pd.DataFrame([[self.baseline]*self.channels], columns=range(self.channels))

    def get_input(self, hist):
        input_row = [self.input_channel_0] + [self.input_other_channels]*(self.channels-1) + [None]
        return pd.DataFrame([input_row], columns=list(range(self.channels))+['State'])

    def get_sustain(self, hist):
        # Get the action selected this iteration
        last_ind = hist.shape[0]-1
        selected_channel = hist.at[last_ind, 'Action']
        # Construct the input df
        input_arr = [self.sustain_other_channels]*self.channels
        input_arr[selected_channel] = self.sustain_selected_channel
        return pd.DataFrame([input_arr], columns=range(self.channels))

    def get_reward(self, hist):
        # Get the action selected this iteration
        last_ind = hist.shape[0]-1
        selected_channel = hist.at[last_ind, 'Action']
        if selected_channel == 0:
            # Get reward for correct action with probability reward_prob
            if random.random() < self.reward_prob:
                return self.correct_reward
            else:
                return self.incorrect_reward
        else:
            # Get reward for incorrect action with probability reward_prob
            if random.random() < self.reward_prob:
                return self.incorrect_reward
            else:
                return self.correct_reward

class EnvironmentFromDF(EnvironmentBase):
    def __init__(self, baseline, input_df, sustain_df, reward_df, channels=2):
        # Validate inputs
        if 'State' not in input_df.columns \
                or any(channel not in input_df.columns for channel in range(channels)):
            raise ValueError("Incorrect input_df columns; must have a column for each channel and a 'State' column")
        if 'Selected' not in sustain_df.columns or 'Other' not in sustain_df.columns:
            raise ValueError("Incorrect sustain_df columns; must have columns 'Selected' and 'Other'")
        if 'Correct' not in reward_df.columns or 'Incorrect' not in reward_df.columns:
            raise ValueError("Incorrect reward_df columns; must have columns 'Correct' and 'Incorrect'")

        self.baseline = baseline # Baseline input FR
        self.input_df = input_df # Should have a row for each iteration and a column for each channel
        # It must also have a 'State' column which contains the index of the correct channel
        self.sustain_df = sustain_df # Should have a row for each iteration and columns 'Selected' and 'Other'
        self.reward_df = reward_df # Should have a row for each iteration and columns 'Correct' and 'Incorrect'
        self.channels = channels

        # Each df can have only one row, in which case it is repeated each iteration
        # Note that if it is not repeated, an error will be thrown when it runs out of rows

    def get_baseline(self, hist):
        return pd.DataFrame([[self.baseline]*self.channels], columns=range(self.channels))

    def get_input(self, hist):
        if self.input_df.shape[0] == 1 or self.input_df.ndim == 1: # If only one row, repeat it
            return self.input_df
        else:
            # Get current index
            last_ind = hist.shape[0]-1
            return self.input_df.loc[last_ind]

    def get_sustain(self, hist):
        # Get the action selected this iteration
        last_ind = hist.shape[0]-1
        selected_channel = hist.at[last_ind, 'Action']
        if self.sustain_df.shape[0] == 1 or self.sustain_df.ndim == 1: # If only one row, repeat it
            # Construct the input df
            input_arr = [self.sustain_df.at[0,'Other']]*self.channels
            input_arr[selected_channel] = self.sustain_df.at[0,'Selected']
        else:
            # Construct the input df
            input_arr = [self.sustain_df.at[last_ind,'Other']]*self.channels
            input_arr[selected_channel] = self.sustain_df.at[last_ind,'Selected']
        return pd.DataFrame([input_arr], columns=range(self.channels))

    def get_reward(self, hist):
        # Get the action selected this iteration
        last_ind = hist.shape[0]-1
        selected_channel = hist.at[last_ind, 'Action']
        correct_channel = hist.at[last_ind, 'State'] # We're assuming that 'State' contains the rewarded action
        if self.reward_df.shape[0] == 1 or self.reward_df.ndim == 1: # If only one row, repeat it
            if correct_channel == selected_channel:
                return self.reward_df.at[0,'Correct']
            else:
                return self.reward_df.at[0,'Incorrect']
        else:
            if correct_channel == selected_channel:
                return self.reward_df.at[last_ind,'Correct']
            else:
                return self.reward_df.at[last_ind,'Incorrect']

