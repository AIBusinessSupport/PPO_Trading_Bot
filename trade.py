import time
import gym
from gym import spaces
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import requests
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from ta import momentum, volatility
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from environment import CryptoTradingEnv


class TextHandler(logging.Handler):
    def _init_(self, outer):
        super()._init_()
        self.outer = outer
        
    def emit(self, record):
        log_entry = self.format(record)
        self.outer.update_log_viewer(log_entry)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

class BTCBotApp:

    def _init_(self, root):

        self.master = root
        self.state_size = 15  # Define according to your features
        self.action_size = 3  # Buy, Hold, Sell
        env = [CryptoTradingEnv(self.data)]
        self.agent = DQNAgent(DummyVecEnv(env), self.state_size, self.action_size)

        self.position = 0
        self.entry_price = 0
        self.margin = 0
        self.accumulated_pnl = 0
        self.order_type = ""
        self.historical_orders = []
        self.PNL_timestamps = []
        self.historic_PNL = []
        self.stop_loss = 50
        self.take_profit = 100
        self.scatter_points = []
        self.symbol = "BTCUSD"
        self.interval = "15m"
        self.limit = 100
        self.current_step = 0
        self.data = self.fetch_kline_data() # Store Kline data here
        self.state = None
        self.kline_data = pd.DataFrame()
        self.simulated_prices = []
        self.kcounter = 0

        self.root = root
        self.root.title("BTC Trading Bot")
        self.root.configure(bg="#111111")

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)


        self.scatter_points = {'buy': [], 'sell': []}
        self.scatter_colors = {'buy': 'green', 'sell': 'red'}
        self.scatter_markers = {'buy': '^', 'sell': 'v'}
        self.scatter_labels = {'buy': 'Buy', 'sell': 'Sell'}  

        self.position_label = ttk.Label(root, text="No Open Position", foreground="white", background="#111111")
        self.position_label.grid(row=0, column=1, padx=20, pady=10)
        accumulated_pnl_label = tk.Label(root, text="Accumulated PNL: $0.00", font=("Helvetica", 16), foreground="green", bg="#111111")
        accumulated_pnl_label.grid(row=9, column=0, padx=20, pady=10, sticky="w")

        self.latest_price_label = ttk.Label(root, text="Latest Price: Loading...", foreground="white", background="#111111")
        self.latest_price_label.grid(row=1, column=1, padx=20, pady=10)
        self.log_viewer_label = tk.Label(self.root, text="Log Viewer", font=("Helvetica", 16), foreground="white", bg="#111111")
        self.log_viewer_label.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky="w")
        
        self.scrollbar = tk.Scrollbar(self.root)
        self.scrollbar.grid(row=4, column=2, sticky="ns")
        
        self.log_viewer = tk.Text(self.root, height=10, width=80, bg="#111111", foreground="white", wrap=tk.WORD, yscrollcommand=self.scrollbar.set)
        self.log_viewer.grid(row=4, column=0, columnspan=2, padx=20, pady=10)
        self.log_viewer.insert(tk.END, "Log Viewer Initialized\n")
        self.log_viewer.config(state=tk.DISABLED)
        
        self.scrollbar.config(command=self.log_viewer.yview)

        self.accumulated_pnl_label = ttk.Label(root, text="Accumulated PNL: $0.00", foreground="white", background="#111111")
        self.accumulated_pnl_label.grid(row=2, column=1, padx=20, pady=10)
        
        self.frame = ttk.Frame(root)
        self.frame.grid(row=0, column=0, rowspan=4, padx=20, pady=10)
        self.fig = Figure(figsize=(9 * 1.25, 14), dpi=100, facecolor='#111111')
        
        self.ax1 = self.fig.add_subplot(511, facecolor='#111111')
        self.ax2 = self.fig.add_subplot(512, sharex=self.ax1, facecolor='#111111')
        self.ax3 = self.fig.add_subplot(513, sharex=self.ax1, facecolor='#111111')
        self.ax4 = self.fig.add_subplot(514, sharex=self.ax1, facecolor='#111111')
        self.ax5 = self.fig.add_subplot(515, sharex=self.ax1, facecolor='#111111')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

        handler = TextHandler(self)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logging.getLogger().addHandler(handler)

        self.update_kline_data()

    def simulate_price_movement(self, open_price, high_price, low_price, close_price, steps=60):
        self.simulated_prices = [open_price]
        for _ in range(steps - 2):
            next_price = random.uniform(low_price, high_price)
            self.simulated_prices.append(next_price)
        self.simulated_prices.append(close_price)
        return self.simulated_prices

    # Your existing functions like calculate_margin_short, close_long_position, etc. will go here.
    # Make sure to remove the "global" keyword and instead refer to instance variables with self.
    def fetch_kline_data(self):
        if os.path.exists("data1.csv"):
            all_data = pd.read_csv("data1.csv")
            all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
            all_data.set_index('timestamp', inplace=True)
            start_idx = self.current_step
            end_idx = self.current_step + 99
            self.kline_data = all_data.iloc[start_idx:end_idx]

            # Simulate the 100th data point
            high = self.kline_data['high'].iloc[-1]
            low = self.kline_data['low'].iloc[-1]
            simulated_close = random.uniform(low, high)

            last_row = pd.DataFrame([self.kline_data.iloc[-1]], columns=self.kline_data.columns)
            last_row['close'] = simulated_close

            self.kline_data = pd.concat([self.kline_data, last_row])
            
            # Simulate the fluctuating price for 60 cycles
            while self.kcounter > 61:
                simulated_price = random.uniform(low, high)
                last_row = pd.DataFrame([self.kline_data.iloc[-1]], columns=self.kline_data.columns)
                last_row['close'] = simulated_price
                self.kline_data = pd.concat([self.kline_data, last_row])
                self.kcounter += self.kcounter
                time.sleep(1)

            if self.kcounter == 60:
                self.kcounter = 0

            # Make 59th data point close to closing price
            close_price = self.kline_data['close'].iloc[-2]
            self.kline_data['close'].iloc[-61] = close_price

            # Make 2nd data point close to opening price
            open_price = self.kline_data['open'].iloc[0]
            self.kline_data['close'].iloc[0] = open_price

            return self.kline_data
        else:
            print("data1.csv does not exist.")
            return None


    def reset(self):
        self.portfolio = 10000.0
        self.current_step = 0
        self.data = self.load_data()
        self.state = self.create_observation()
        return self.state

    def step(self):
        
        self.current_step += 1
        self.state = self.create_observation()
        reshaped_state = np.reshape(self.state, [1, self.state_size]) 
        done = False
        action = self.agent.act(reshaped_state)
        reward = self.accumulated_pnl

        if action == 1:
            self.open_long_position(self.kline_data['sma5'].iloc[99])
            self.historical_orders.append((self.kline_data.index[99], 'BUY', self.kline_data['sma5'].iloc[99]))
            
        elif action == 2:
            self.open_short_position(self.kline_data['sma5'].iloc[99])
            self.historical_orders.append((self.kline_data.index[99], 'SELL', self.kline_data['sma5'].iloc[99]))

        if self.current_step >= len(self.data) - 1:
            done = True

        self.historic_PNL.append(self.accumulated_pnl)
        self.PNL_timestamps.append(datetime.utcnow())
        print(self.current_step)
        return self.state, reward, done, {}

    def render(self, episode, step, reward):
        print(f"Episode: {episode} - Step: {step} - Reward: {reward}")

    def load_data(self):
        return np.random.rand(100, 10)

    def create_observation(self):
        # Check if current_step is within the valid range of indices
        if 0 <= self.current_step < len(self.data):
            observation = self.data.iloc[self.current_step]  # Use iloc to access DataFrame by integer index
            return observation
        else:
            # Handle the case when current_step is out of bounds
            return None  # You can return None or handle this case as needed


    def place_order(self, symbol, side, quantity, price):
        # This is a mock function. In a real-world scenario, you'd call the Binance API here.
        logging.info(f"Placed {side} order for {quantity} of {symbol} at ${price:.2f}")

    def open_short_position(self, price):
        self.position = 4
        self.entry_price = price
        self.margin = 0
        self.order_type = "Short"
        self.position_label.config(text=f"Position Size: 2000 BTC\nEntry Price: {self.entry_price:.4f} BTC")
        self.place_order('BTCUSDT', 'SELL', self.position, self.entry_price)
        logging.info("Opened short position.")

    def open_long_position(self, price):
        self.position = 4
        self.entry_price = price
        self.margin = 0
        self.order_type = "Long"
        self.position_label.config(text=f"Position Size: 2000 BTC\nEntry Price: {self.entry_price:.4f} BTC")
        self.place_order('BTCUSDT', 'BUY', self.position, self.entry_price)
        logging.info("Opened long position.")


    def calculate_margin_short(self, price):
        self.margin = (self.entry_price - price) * self.position
        self.position_label.config(text=f"Position Size: {self.position} BTC\nEntry Price: {self.entry_price:.4f} BTC\nMargin: ${self.margin:.4f}")
        logging.info(f"Margin updated: {self.margin:.2f}")

    def calculate_margin_long(self, price):
        self.margin = (price - self.entry_price) * self.position
        self.position_label.config(text=f"Position Size: {self.position} BTC\nEntry Price: {self.entry_price:.4f} BTC\nMargin: ${self.margin:.4f}")
        logging.info(f"Margin updated: {self.margin:.2f}")

    def close_long_position(self, price, reason):
        self.accumulated_pnl += self.margin
        self.margin = 0
        self.place_order('BTCUSDT', 'SELL', self.position, price)
        self.position = 0
        self.entry_price = 0
        self.order_type = ""
        self.position_label.config(text=f"Position Closed. Reason: {reason}")
        
        logging.info("Closed long position.")

    def close_short_position(self, price, reason):
        self.accumulated_pnl += self.margin
        self.margin = 0
        self.place_order('BTCUSDT', 'BUY', self.position, price)
        self.position = 0
        self.entry_price = 0
        self.order_type = ""
        self.position_label.config(text=f"Position Closed. Reason: {reason}")
        logging.info("Closed short position.")
        
    def update_log_viewer(self, message):
        self.log_viewer.config(state=tk.NORMAL)
        self.log_viewer.insert(tk.END, message + "\n")
        self.log_viewer.config(state=tk.DISABLED)
        self.log_viewer.yview(tk.END)

    def update_kline_data(self):
        self.kline_data = self.fetch_kline_data()
        earliest_kline_time = self.kline_data.index.min()
        latest_kline_time = self.kline_data.index.max()
        self.data = self.fetch_kline_data()
        
        latest_BTC_price = self.kline_data.iloc[99].iloc[0]
        print(latest_BTC_price)
        buy_volume = self.kline_data['taker_buy_base_asset_volume'].astype(float)
        sell_volume = self.kline_data['volume'].astype(float) - buy_volume*.5

        buy_times = [order_time for order_time, side, price in self.historical_orders if side == 'BUY']
        buy_prices = [price for order_time, side, price in self.historical_orders if side == 'BUY']
        sell_times = [order_time for order_time, side, price in self.historical_orders if side == 'SELL']
        sell_prices = [price for order_time, side, price in self.historical_orders if side == 'SELL']
        filtered_buy_times = [order_time for order_time, side, price in self.historical_orders if side == 'BUY' and earliest_kline_time <= order_time <= latest_kline_time]
        filtered_buy_prices = [price for order_time, side, price in self.historical_orders if side == 'BUY' and earliest_kline_time <= order_time <= latest_kline_time]
        filtered_sell_times = [order_time for order_time, side, price in self.historical_orders if side == 'SELL' and earliest_kline_time <= order_time <= latest_kline_time]
        filtered_sell_prices = [price for order_time, side, price in self.historical_orders if side == 'SELL' and earliest_kline_time <= order_time <= latest_kline_time]
        filtered_PNL_timestamps = [time for time in self.PNL_timestamps if earliest_kline_time <= time <= latest_kline_time]
        filtered_historic_PNL = [pnl for time, pnl in zip(self.PNL_timestamps, self.historic_PNL) if earliest_kline_time <= time <= latest_kline_time]

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.kline_data['timestamp'] = mdates.date2num(self.kline_data.index.to_pydatetime())
        candlestick_data = [tuple(x) for x in self.kline_data[['timestamp', 'open', 'high', 'low', 'close']].values]
        self.ax1.xaxis_date()
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        candlestick_ohlc(self.ax1, candlestick_data, width=0.00125*0.5, colorup='orange', colordown='grey')

        self.ax1.plot(self.kline_data.index, self.kline_data['sma5'], label='SMA(5)', color='orange')
        self.ax1.plot(self.kline_data.index, self.kline_data['bollinger_mavg'], color='white', linestyle=':')
        #self.ax1.plot(self.kline_data.index, self.kline_data['bollinger_hband'], color='grey', linestyle='--')
        #self.ax1.plot(self.kline_data.index, self.kline_data['bollinger_lband'], color='grey', linestyle='--')
        self.ax1.scatter(filtered_buy_times, filtered_buy_prices, color='green', label='Buy Orders', zorder=5, marker='^')
        self.ax1.scatter(filtered_sell_times, filtered_sell_prices, color='red', label='Sell Orders', zorder=5, marker='v')
        self.ax1.set_ylabel('Price', color='white')
        self.ax1.tick_params(axis='both', colors='white')
        self.ax1.legend()

        self.ax2.xaxis_date()
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.ax2.plot(filtered_PNL_timestamps, filtered_historic_PNL, color='white', label='Cumulative P&L')
        self.ax2.set_ylabel('PNL', color='white')
        self.ax2.tick_params(axis='both', colors='white')
        self.ax2.legend()

        self.ax3.plot(self.kline_data.index, self.kline_data['macd'], label='MACD', color='cyan')
        self.ax3.plot(self.kline_data.index, self.kline_data['macd_signal'], label='Signal', color='magenta')
        self.ax3.set_ylabel('MACD', color='white')
        self.ax3.tick_params(axis='both', colors='white')
        self.ax3.legend()
        self.ax4.plot(self.kline_data.index, self.kline_data['rsi'], label='RSI', color='orange')
        self.ax4.axhline(75, color='grey', linestyle='--')
        self.ax4.axhline(25, color='grey', linestyle='--')
        self.ax4.tick_params(axis='both', colors='white')
        self.ax4.plot(self.kline_data.index, self.kline_data['stoch_rsi'], label='Stoch RSI', color='yellow')
        self.ax4.set_ylabel('RSI and Stoch RSI', color='white')
        self.ax4.tick_params(axis='both', colors='white')
        self.ax4.legend()

        self.ax5.bar(self.kline_data.index, buy_volume, label='Buy Volume', color='green', alpha=0.7, width=0.00125*.5)
        self.ax5.bar(self.kline_data.index, sell_volume, label='Sell Volume', color='red', alpha=0.7, width=0.00125*.5, bottom=buy_volume)
        self.ax5.set_ylabel('Volume', color='white')
        self.ax5.tick_params(axis='both', colors='white')
        self.ax5.legend()

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.tick_params(axis='x', labelbottom=False)

        title = f"BTC Trading Bot - Current Price: ${latest_BTC_price:.2f} BTC"
        if self.position != 0:
            title += f" - Margin: ${self.margin:.2f}"
        self.fig.suptitle(title, fontsize=16, color='white')
        self.latest_price_label.config(text=f"Latest Price: ${latest_BTC_price:.2f} BTC")
        self.accumulated_pnl_label.config(text=f"Accumulated PNL: ${self.accumulated_pnl:.2f}", foreground="green" if self.accumulated_pnl >= 0 else "red")
        self.canvas.draw()
        self.step()
        root.after(1000, self.update_kline_data)

class MyPolicy(ActorCriticPolicy):
    def __init__(self, model):
        super(MyPolicy, self).__init__()
        self.model = model

class DQNAgent:
    def _init_(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.n_steps = 2000
        self.verbose = 1
        self.init_model = self.get_RLModel(policy=MyPolicy(self._build_model()), n_steps=self.n_steps, batch_size=self.batch_size, 
                                      ent_coef=self.epsilon_min, lr=self.learning_rate, verbose=self.verbose)

        self.model = self.train_model(self.init_model, 'PPO', 5000)
        
    def get_RLModel(self, policy, n_steps, batch_size, ent_coef, lr, verbose):
        return PPO(policy=policy, env=self.env, verbose=verbose, gamma=self.gamma,
                   learning_rate=lr, n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef)
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    @staticmethod
    def train_model(
        model, tb_log_name, total_timesteps=5000
    ):  # this function is static method, so it can be called without creating an instance of the class
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        return model
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the BTCBotApp instance and start the main loop
root = tk.Tk()
logging.basicConfig(level=logging.INFO)
bot = BTCBotApp(root)

bot.update_kline_data()
root.mainloop()