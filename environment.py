
import psutil
import os
import gym
from gym import spaces
from gym.utils import seeding
from enum import Enum
import numpy as np 

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import stable_baselines3 as sb3

import matplotlib.pyplot as plt

import pandas_ta as ta

import socket
#import math
import random


class Trade_Definitions():
    # Definitions
    PositionFlat = 0
    PositionLong = 1
    PositionShort = -1
    Open_On_Open = 'OpenOnOpen'
    Open_On_Close = 'OpenOnClose'
    Close_On_Open = 'CloseOnOpen'
    Close_On_Close = 'CloseOnClose'
    


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.verbose = 0

        self.max_bars = 999999
        self.sl_percent = -999999
        self.open_on=Trade_Definitions.Open_On_Close
        self.close_on=Trade_Definitions.Close_On_Close
        self.trade_amount=100000
        self.commission=-50

        self.prices, self.signal_features, self.extra_fields = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(3)      # Short, Flat, Long = 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_bar = self.window_size
        self._end_bar = len(self.prices) - 1
        self._done = None
        self._last_current_bar = None
        #self._last_ins = None
        #self._last_state_change_bar = None
        self.history = {}
        self._first_rendering = True

        # logic
        self._current_state = Trade_Definitions.PositionFlat
        self._current_state_history = (self.window_size * [None]) + [self._current_state]

        self.clsTrade = Trade(
            instrument='dummy', 
            open_on=self.open_on, 
            close_on=self.close_on, 
            trade_amount=self.trade_amount, 
            commission=self.commission, 
            verbose=0)

        # Trade stats
        self.Trade_Stats_Total = {
            'Total_Trades':  0,
            'ClosedPnL_Total':  0,
            'ClosedPnL_Avg':  0
        }

            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._last_current_bar = random.randint( self._start_bar, self.frame_bound[1]-self.window_size )
        self._current_state = Trade_Definitions.PositionFlat

        self.clsTrade.instrument = self.extra_fields.iloc[self._last_current_bar]['Ins']
        self.clsTrade.open_on = self.open_on
        self.clsTrade.close_on = self.close_on
        self.clsTrade.trade_amount = self.trade_amount
        self.clsTrade.commission = self.commission
            
        self.clsTrade.reset()

        return self._get_observation()


    def step(self, action):
        self._done = False
        self._last_current_bar += 1
        
        if self._last_current_bar >= self._end_bar:
            # End of data or end of trade. Done
            self._done = True

        self._done = self.clsTrade.step( action-1, self.extra_fields.iloc[self._last_current_bar])

        if self._done:
            # Trade ended
            reward_step_pnl = self.clsTrade.ClosedPnL
            self.Trade_Stats_Total['Total_Trades'] += 1
            self.Trade_Stats_Total['ClosedPnL_Total'] += self.clsTrade.ClosedPnL
            self.Trade_Stats_Total['ClosedPnL_Avg'] = self.Trade_Stats_Total['ClosedPnL_Total'] / self.Trade_Stats_Total['Total_Trades']

        elif self.clsTrade.position != Trade_Definitions.PositionFlat:
            # Ongoing Trade
            reward_step_pnl = self.clsTrade.OpenPnl
        else:
            # No Trade
            reward_step_pnl = 0

        if self.clsTrade.StatsBarsInTrade > self.max_bars:
            # Punish when trade goes on for to long
            nExpVar = self.clsTrade.StatsBarsInTrade/2
            reward_step_pnl += -(nExpVar**nExpVar)
        
        if (self.clsTrade.OpenPnl/self.trade_amount) < self.sl_percent:
            nExpVar = self.clsTrade.StatsBarsInTrade/2
            reward_step_pnl += -(nExpVar**nExpVar)
            #reward_step_pnl += -99999

        info = {
            'action-1'                      : action-1,
            'last_current_bar'              : self._last_current_bar,
            'Instrument'                    : self.clsTrade.instrument,
            'Date'                          : self.clsTrade.GetCurrentDate(),
            'Current_state'                 : self.clsTrade.position,
            'trade_open_pnl'                : round(self.clsTrade.OpenPnl,2),
            'trade_closed_pnl'              : round(self.clsTrade.ClosedPnL,2),
        }

        if self.verbose==1:
            print(info)

        self._current_state_history.append(self._current_state)

        # Add extra information to observation
        self.signal_features[self._last_current_bar][-4] =  self.clsTrade.StatsBarsInTrade
        self.signal_features[self._last_current_bar][-3] =  self.clsTrade.position
        self.signal_features[self._last_current_bar][-2] =  self.clsTrade.OpenPnl
        self.signal_features[self._last_current_bar][-1] =  self.clsTrade.ClosedPnL

        
        observation = self._get_observation()
        
        self._update_history(info)

        return observation, reward_step_pnl, self._done, info
    

    def _get_observation(self):
        return self.signal_features[(self._last_current_bar-self.window_size+1):self._last_current_bar+1]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_current_state(current_state, tick):
            color = None
            if current_state == -1:
                color = 'red'
            elif current_state == 1:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_current_state = self._current_state_history[self._start_bar]
            _plot_current_state(start_current_state, self._start_bar)

        _plot_current_state(self._current_state, self._last_current_bar)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_pnl + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._current_state_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._current_state_history[i] == -1:
                short_ticks.append(tick)
            elif self._current_state_history[i] == 1:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_pnl + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _get_extra_fields(self,cField):
        
        if cField=='Date': cValue = self.extra_fields[self._last_current_bar][0]
        if cField=='Ins': cValue = self.extra_fields[self._last_current_bar][1]
        if cField=='Open': cValue = self.extra_fields[self._last_current_bar][2]
        if cField=='High': cValue = self.extra_fields[self._last_current_bar][3]
        if cField=='Low': cValue = self.extra_fields[self._last_current_bar][4]
        if cField=='Close': cValue = self.extra_fields[self._last_current_bar][5]
        if cField=='Volume': cValue = self.extra_fields[self._last_current_bar][6]

        return cValue
    
    
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        lextraCols = ['Date','Ins','Open','High','Low','Close','Volume']
        #extra_fields = self.df.loc[:, lextraCols].to_numpy()[start:end]
        # Changed extra_fields to Pandas instead of Numpy
        extra_fields = self.df.iloc[start:end][lextraCols]

        prices = self.df.loc[:, 'Close'].to_numpy()[start:end]   # Trade on Close
        
        
        # Add extra field to signal_features. Theese will also be added to observation
        self.df['StatsBarsInTrade'] = 0
        self.df['position'] = 0
        self.df['OpenPnl'] = 0.0
        self.df['ClosedPnL'] = 0.0
        
        # Remove OHLC as features and add Position and Current PnL
        lCols = self.df.columns.to_list()
        lCols.remove('Date')
        lCols.remove('Ins')
        lCols.remove('Open')
        lCols.remove('High')
        lCols.remove('Low')
        lCols.remove('Close')
        lCols.remove('Volume')
        
        signal_features = self.df.loc[:, lCols].to_numpy()[start:end]
        
        return prices, signal_features, extra_fields


    def _calculate_stats(self):
        pass        



class TensorboardCallback(BaseCallback):
    """ Logs the net change in cash between the beginning and end of each epoch/run. """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        dStatsTotal = self.training_env.get_attr('Trade_Stats_Total')[0]

        self.logger.record("Trade_Metrics/ClosedPnL_Avg", dStatsTotal['ClosedPnL_Avg'])
        
        return True


class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, num_features):
        # Define observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_features,), dtype=np.float32)
        
        # Define action space
        self.action_space = spaces.Discrete(3)
        
        # Define reward range
        self.reward_range = (0, 1)
        
        # Store historical price data
        self.df = df
        
        # Set initial step to 0
        self.current_step = 0
        
    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Get next observation
        obs = self._get_obs()
        
        # Check if done
        done = self.current_step == len(self.df) - 1
        
        return obs, reward, done, {}
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        return self._get_obs()
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def _get_obs(self):
        # Get the observation at the current time step
        obs = self.df.iloc[self.current_step].values
        return obs
    
    def _calculate_reward(self, action):
        # Calculate the reward for the current time step and action
        pass
