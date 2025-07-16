import gymnasium as gym
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
import logging
from collections import deque

import config


class PortfolioEnv(gym.Env):
    """Portfolio allocation environment for cryptocurrency trading using Gymnasium interface
    
    Supports both fixed portfolio (legacy) and variable portfolio sizes with masking.
    """

    def __init__(self, use_variable_portfolio=None, reward_type="simple"):
        super(PortfolioEnv, self).__init__()
        
        # Use config setting or override
        self.use_variable_portfolio = (
            use_variable_portfolio if use_variable_portfolio is not None 
            else config.USE_VARIABLE_PORTFOLIO
        )
        
        # Set reward type for different reward calculations
        self.reward_type = reward_type
        print(f"🎯 Portfolio Environment initialized with reward_type: {reward_type}")
        
        if self.use_variable_portfolio:
            self._setup_variable_portfolio()
        else:
            self._setup_fixed_portfolio()
        
        # Initialize environment state
        self.reset()

    def _setup_variable_portfolio(self):
        """Setup observation and action spaces for variable portfolio size."""
        
        # Action space: portfolio weights for cash + max possible coins
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(config.MAX_COINS + 1,),  # +1 for cash
            dtype=np.float64
        )
        
        # Observation space: Dict with observations and mask
        self.observation_space = gym.spaces.Dict({
            'observations': gym.spaces.Box(
                low=np.array(config.OBS_COLS_MIN_VARIABLE), 
                high=np.array(config.OBS_COLS_MAX_VARIABLE), 
                shape=(config.MAX_OBSERVATION_DIM,), 
                dtype=np.float64
            ),
            'mask': gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.MAX_COINS,),  # Mask for coins (not cash)
                dtype=np.float64
            )
        })
        
        print(f"🔧 Variable portfolio setup:")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: Dict with {config.MAX_OBSERVATION_DIM} features + {config.MAX_COINS} mask")

    def _setup_fixed_portfolio(self):
        """Setup observation and action spaces for fixed portfolio size (legacy)."""
        
        # Define action space: portfolio weights (including cash) - continuous values between 0 and 1
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(len(config.COINS)+1,), 
            dtype=np.float64
        )
        
        # Define observation space: scaled market features
        self.observation_space = gym.spaces.Box(
            low=np.array(config.OBS_COLS_MIN), 
            high=np.array(config.OBS_COLS_MAX), 
            shape=(len(config.OBS_COLS),), 
            dtype=np.float64
        )
        
        print(f"🔧 Fixed portfolio setup (legacy):")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space}")

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Select coins for this episode
        if self.use_variable_portfolio:
            self.episode_coins = config.get_random_coin_subset()
            self.n_episode_coins = len(self.episode_coins)
            print(f"🎲 Episode coins: {self.episode_coins} ({self.n_episode_coins} coins)")
        else:
            self.episode_coins = config.COINS
            self.n_episode_coins = len(self.episode_coins)
        
        self.memory_return = pd.DataFrame(columns=[t+"_close" for t in self.episode_coins])
        self._episode_ended = False
        self.index = 0
        self.time_delta = pd.Timedelta(5, unit='m')
        self.init_cash = 1000
        self.current_cash = self.init_cash
        self.current_value = self.init_cash
        self.previous_price = {}
        self.old_dict_coin_price_1 = {}
        self.old_dict_coin_price_2 = {}

        # --- FIX: Randomize initial allocation instead of starting with 100% cash ---
        # This forces the agent to learn to manage an active portfolio from step 1
        # and prevents it from getting stuck in a "do nothing" local optimum.
        # We use a Dirichlet distribution to ensure the initial weights sum to 1.
        if self.use_variable_portfolio:
            active_positions = self.n_episode_coins + 1  # +1 for cash
            dirichlet_alpha = np.ones(active_positions)
            random_initial_split = np.random.dirichlet(dirichlet_alpha)
            
            self.money_split_ratio = np.zeros(config.MAX_COINS + 1)
            self.money_split_ratio[:active_positions] = random_initial_split
            print(f"💰 Randomized initial allocation (variable): {[f'{x:.2%}' for x in random_initial_split]}")
        else:
            active_positions = len(self.episode_coins) + 1  # +1 for cash
            dirichlet_alpha = np.ones(active_positions)
            self.money_split_ratio = np.random.dirichlet(dirichlet_alpha)
            print(f"💰 Randomized initial allocation (fixed): {[f'{x:.2%}' for x in self.money_split_ratio]}")

        self.df = pd.read_csv(config.FILE)
        self.scaler = preprocessing.StandardScaler()
        
        self.df["date"] = self.df["date"].apply(lambda x: pd.Timestamp(x, unit='s', tz='US/Pacific'))
        
        # Filter for episode coins only
        self.df = self.df[self.df["coin"].isin(self.episode_coins)].sort_values("date")
        self.scaler.fit(self.df[config.SCOLS].values)
        self.df = self.df.reset_index(drop=True)

        self.max_index = self.df.shape[0]

        # Fixed episode slicing logic to match successful implementation approach
        # Keep total data rows constant (like successful implementation uses 1500 rows)
        # This means: 1 coin = 1500 steps, 2 coins = 750 steps, 3 coins = 500 steps
        total_data_rows = config.EPISODE_LENGTH  # 1500 rows like successful implementation
        time_steps_per_episode = total_data_rows // self.n_episode_coins
        
        if self.max_index > total_data_rows + 3:
            max_start_point = self.max_index - total_data_rows - 3
            # Ensure start_point is aligned with the start of a timestamp block
            start_timestamp = np.random.randint(0, max_start_point // self.n_episode_coins)
            start_point = start_timestamp * self.n_episode_coins
        else:
            # If dataset is too small, start from the beginning
            start_point = 0
            
        end_point = start_point + total_data_rows
        self.df = self.df.loc[start_point:end_point+2].reset_index(drop=True)

        self.df = self.df.reset_index(drop=True)

        self.init_time = self.df.loc[0, "date"]
        self.current_time = self.init_time
        self.dfslice = self.df[(self.df["coin"].isin(self.episode_coins))&(self.df["date"]>=self.current_time)&(self.df["date"]<self.current_time+pd.Timedelta(5, unit='m'))].copy().drop_duplicates("coin")

        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
        self.previous_value = self.current_value
        self.current_stock_money_distribution, self.current_value = self.calculate_money_from_num_stocks()
        
        # Update money split ratio based on calculated distributions (like successful implementation)
        if self.use_variable_portfolio:
            # Only update the positions for active coins
            for i, coin in enumerate(self.episode_coins):
                self.money_split_ratio[i+1] = self.current_stock_money_distribution[i+1] / self.current_value if self.current_value > 0 else 0
            self.money_split_ratio[0] = self.current_cash / self.current_value if self.current_value > 0 else 1
        else:
            self.money_split_ratio = self.normalize_money_dist()
        
        self.step_reward = 0
        
        # Initialize actual allocation tracking (NEW - for ROOT CAUSE #1 fix)
        self.actual_money_split_ratio = self.money_split_ratio.copy() if hasattr(self, 'money_split_ratio') else None
        
        # Initialize transaction cost tracking for Net Return calculation
        self.previous_money_split_ratio = self.money_split_ratio.copy() if hasattr(self, 'money_split_ratio') else None
        self.total_transaction_costs = 0.0
        
        # Initialize long-term performance tracking for bonus calculation
        self.portfolio_value_history = [self.current_value]  # Track portfolio values over time
        self.long_term_bonus = 0.0  # Current long-term bonus component
        
        # --- NEW: History for structured credit assignment ---
        self.allocation_history = []
        self.price_history = []
        WINDOW = config.LONG_TERM_LOOKBACK
        self.return_history = deque(maxlen=config.LONG_TERM_LOOKBACK)
        
        # Get observations
        scaled_output = self.get_observations()
        
        if self.use_variable_portfolio:
            observation, mask = self._create_variable_observation(scaled_output)
            info = {
                "state": "state",
                "money_split": self.money_split_ratio,
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "episode_coins": self.episode_coins,
                "mask": mask,
                "n_active_coins": self.n_episode_coins,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0)
            }
            obs_dict = {"observations": observation, "mask": mask}
        else:
            observation = scaled_output[config.OBS_COLS].values.flatten()
            info = {
                "state": "state",
                "money_split": self.money_split_ratio,
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "scaled_output": scaled_output,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0)
            }
            obs_dict = observation
        
        # Fixed episode termination logic to match successful implementation approach
        # Episodes run for total_data_rows // n_coins time steps
        time_steps_per_episode = config.EPISODE_LENGTH // self.n_episode_coins
        self._episode_ended = True if self.index == time_steps_per_episode else False
        
        return obs_dict, info

    def step(self, action):
        """Execute one step in the environment"""
        
        if self._episode_ended:
            # If episode ended, reset environment
            return self.reset()
        
        # 🎯 ESSENTIAL: Show agent's action
        print(f"🎯 Agent Action: {action}")
        
        # Handle action based on portfolio type
        if self.use_variable_portfolio:
            action = self._apply_action_mask(action)
        
        # Store previous allocation for transaction cost calculation
        if hasattr(self, 'money_split_ratio'):
            self.previous_money_split_ratio = self.money_split_ratio.copy()
        
        # Normalize action to ensure it sums to 1
        action = np.array(action)  # Ensure action is numpy array
        if sum(action) <= 1e-3:
            print(f"⚠️  Invalid action sum ({sum(action):.6f}), using safe default")
            if self.use_variable_portfolio:
                self.money_split_ratio = np.zeros(config.MAX_COINS + 1)
                self.money_split_ratio[0] = 1  # All cash if invalid action
            else:
                self.money_split_ratio = np.array([1/len(action) for _ in action])
        else:
            if self.use_variable_portfolio:
                # Only update active positions
                normalized_action = action / sum(action)
                self.money_split_ratio = np.zeros(config.MAX_COINS + 1)
                self.money_split_ratio[0] = normalized_action[0]  # Cash
                for i, coin in enumerate(self.episode_coins):
                    self.money_split_ratio[i+1] = normalized_action[i+1]
            else:
                self.money_split_ratio = action / sum(action)

        # 📊 ESSENTIAL: Show target allocation
        print(f"📊 Target Allocation: {[f'{x:.3f}' for x in self.money_split_ratio]}")

        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
        
        self.step_time()
        self.index += 1

        # Get observations
        scaled_output = self.get_observations()
        
        if self.use_variable_portfolio:
            observation, mask = self._create_variable_observation(scaled_output)
            info = {
                "state": "state",
                "money_split": self.money_split_ratio,  # Now represents actual allocation after price movements
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "episode_coins": self.episode_coins,
                "mask": mask,
                "n_active_coins": self.n_episode_coins,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0)
            }
            obs_dict = {"observations": observation, "mask": mask}
        else:
            observation = scaled_output[config.OBS_COLS].values.flatten()
            info = {
                "state": "state",
                "money_split": self.money_split_ratio,  # Now represents actual allocation after price movements
                "share_num": self.current_stock_num_distribution,
                "value": self.current_value,
                "time": self.current_time,
                "reward": self.step_reward,
                "scaled_output": scaled_output,
                "reward_type": self.reward_type,
                "total_transaction_costs": getattr(self, 'total_transaction_costs', 0.0),
                "long_term_bonus": getattr(self, 'long_term_bonus', 0.0)
            }
            obs_dict = observation

        reward = info["reward"]
        
        # 💰 ESSENTIAL: Show results
        print(f"💰 Portfolio: ${self.current_value:.2f} | Reward: {reward:.4f} | Actual: {[f'{x:.3f}' for x in self.money_split_ratio]}")
        
        # Check if episode is done (use correct episode length matching successful implementation)
        # Episodes run for EPISODE_LENGTH // n_coins time steps (constant data usage)
        time_steps_per_episode = config.EPISODE_LENGTH // self.n_episode_coins
        terminated = self.index >= time_steps_per_episode
        truncated = False  # Add truncation logic if needed
        
        if terminated:
            reward = 0  # Set final reward to 0 as in original
            episode_return = (self.current_value - config.INITIAL_CASH) / config.INITIAL_CASH
            
            if self.reward_type == "TRANSACTION_COST":
                total_costs_pct = (getattr(self, 'total_transaction_costs', 0.0) / config.INITIAL_CASH) * 100
                print(f"🏁 Episode Complete: {self.index} steps, Final Value: ${self.current_value:.2f}")
                print(f"📊 Episode Return: {episode_return:.2%}, Total Transaction Costs: ${getattr(self, 'total_transaction_costs', 0.0):.2f} ({total_costs_pct:.2f}%)")
                print(f"💰 Net Episode Performance: {episode_return:.2%} - {total_costs_pct:.2f}% = {episode_return - total_costs_pct/100:.2%}")
            else:
                print(f"🏁 Episode Complete: {self.index} steps, Final Value: ${self.current_value:.2f}")
                print(f"📊 Episode Return: {episode_return:.2%}")
            
        try:
            return obs_dict, reward, terminated, truncated, info
        except Exception as e:
            print("ERRORRRRRR!!!!!!!!!!!!!!!!")
            print("Observation:", obs_dict)
            print("Reward:", reward)
            print("Step reward, current value, previous value:", self.step_reward, self.current_value, self.previous_value)
            print("Current stock money distribution:", self.current_stock_money_distribution)
            print("Current stock num distribution:", self.current_stock_num_distribution)
            print("Action:", action)
            print("Index:", self.index)
            print("Dfslice:", self.dfslice)
            print("Current time:", self.current_time)
            print("Money split ratio:", self.money_split_ratio)
            print("Episode coins:", self.episode_coins)
            print("Exception:", e)
            
            raise ValueError(f"Error in step function: {e}")

    def _apply_action_mask(self, action):
        """Apply masking to actions for variable portfolio."""
        masked_action = np.zeros(config.MAX_COINS + 1)
        masked_action[0] = action[0]  # Cash is always active
        
        # Only copy actions for active coins
        for i, coin in enumerate(self.episode_coins):
            if i + 1 < len(action):
                masked_action[i + 1] = action[i + 1]
        
        return masked_action

    def _create_variable_observation(self, scaled_output):
        """Create padded observation and mask for variable portfolio."""
        # Create mask: 1 for active coins, 0 for inactive
        mask = np.zeros(config.MAX_COINS)
        mask[:self.n_episode_coins] = 1.0
        
        # Create padded observation
        observation = np.zeros(config.MAX_OBSERVATION_DIM)
        
        if not scaled_output.empty:
            # Fill in features for active coins
            for i, coin in enumerate(self.episode_coins):
                start_idx = i * config.FEATURES_PER_COIN
                end_idx = (i + 1) * config.FEATURES_PER_COIN
                
                coin_features = []
                for scol in config.SCOLS:
                    col_name = f"{coin}_{scol}"
                    if col_name in scaled_output.columns:
                        coin_features.extend(scaled_output[col_name].values)
                    else:
                        coin_features.extend([0.0])  # Pad missing features
                
                # Ensure we have exactly FEATURES_PER_COIN features
                coin_features = coin_features[:config.FEATURES_PER_COIN]
                while len(coin_features) < config.FEATURES_PER_COIN:
                    coin_features.append(0.0)
                
                observation[start_idx:end_idx] = coin_features
        
        return observation, mask

    def step_time(self):
        """Advance time and update portfolio values"""
        self.current_time += self.time_delta
        self.dfslice = self.df[(self.df["coin"].isin(self.episode_coins))&(self.df["date"]>=self.current_time)&(self.df["date"]<self.current_time+pd.Timedelta(5, unit='m'))].copy().drop_duplicates("coin")
        self.previous_value = self.current_value
        
        # Recalculate portfolio value based on new prices and existing share distribution
        self.current_stock_money_distribution, self.current_value = self.calculate_money_from_num_stocks()
        
        # ✅ CORRECTED LOGIC: Calculate the new money split ratio *after* price changes,
        # but before calculating costs. This is the new "actual" allocation.
        new_money_split_ratio = self.normalize_money_dist()

        # Calculate transaction costs for Net Return (Profit minus Costs)
        transaction_cost = 0.0
        volatility_penalty = 0.0 # Initialize volatility penalty
        if (self.reward_type == "TRANSACTION_COST" and 
            hasattr(self, 'previous_money_split_ratio') and 
            self.previous_money_split_ratio is not None):
            
            # The agent's intended action is in `self.money_split_ratio`. The actual
            # allocation from the *previous* step is `self.previous_money_split_ratio`.
            # The change is the difference between the agent's new target and what was
            # actually there before.
            allocation_changes = np.sum(np.abs(np.array(self.money_split_ratio) - np.array(self.previous_money_split_ratio)))
            
            traded_volume_fraction = allocation_changes / 2.0
            
            # Transaction cost = traded volume * portfolio value * cost rate
            transaction_cost = traded_volume_fraction * self.previous_value * config.TRANSACTION_COST
            
            # --- NEW: Volatility Penalty ---
            volatility_penalty = config.VOLATILITY_PENALTY_WEIGHT * (traded_volume_fraction ** 2)
            
            # Debug output for transaction costs
            if traded_volume_fraction > 0.001:
                cost_as_pct_of_value = (transaction_cost / self.previous_value * 100) if self.previous_value > 1e-9 else 0
                print(
                    f"💸 Transaction: Traded {traded_volume_fraction:.2%} of portfolio. "
                    f"Cost: ${transaction_cost:.2f} ({cost_as_pct_of_value:.4f}%)"
                )
        
        # ✅ CRITICAL FIX: The new state of the environment is the `new_money_split_ratio`
        # which reflects the allocation after price drift. This becomes the basis for the next action.
        self.money_split_ratio = new_money_split_ratio
        
        # ✅ LONG-TERM PERFORMANCE TRACKING: Update portfolio value history
        self.portfolio_value_history.append(self.current_value)
        
        # Keep only the required history length for efficiency
        max_history_length = config.LONG_TERM_LOOKBACK + 1
        if len(self.portfolio_value_history) > max_history_length:
            self.portfolio_value_history = self.portfolio_value_history[-max_history_length:]
        
        # Calculate long-term performance bonus: r_t' = r_t + λ_long * 1/N * (V_t - V_{t-N}) / V_{t-N}
        self.long_term_bonus = 0.0
        if (config.LONG_TERM_BONUS_ENABLED and 
            len(self.portfolio_value_history) > config.LONG_TERM_LOOKBACK):
            
            # Get portfolio value N steps ago
            v_t_minus_n = self.portfolio_value_history[-(config.LONG_TERM_LOOKBACK + 1)]
            v_t = self.current_value
            
            # Calculate long-term bonus: λ_long * (1/N) * (V_t - V_{t-N}) / V_{t-N}
            if v_t_minus_n > 1e-9:  # Avoid division by zero
                long_term_return = (v_t - v_t_minus_n) / v_t_minus_n
                self.long_term_bonus = config.LONG_TERM_LAMBDA * long_term_return #* (1.0 / config.LONG_TERM_LOOKBACK) 
                
                # Debug output for long-term bonus
                if abs(self.long_term_bonus) > 0.001:  # Only log significant bonuses
                    print(f"📈 Long-term Bonus: λ={config.LONG_TERM_LAMBDA:.2f} × "
                          f"({v_t:.2f} - {v_t_minus_n:.2f})/{v_t_minus_n:.2f} = "
                          f"{self.long_term_bonus:.4f}")

        # ✅ REFACTOR: Calculate and store step return history REGARDLESS of reward type.
        # This is crucial for the Sortino scaling calculation.
        if self.previous_value > 1e-9:
            step_return = (self.current_value - self.previous_value) / self.previous_value
        else:
            step_return = 0.0
        self.return_history.append(step_return)
        
        # --- NEW: Store history for structured rewards ---
        if self.reward_type == "STRUCTURED_CREDIT":
            # The allocation to store is the one the agent decided on for this step
            self.allocation_history.append(self.last_target_allocation)
            if len(self.allocation_history) > config.LONG_TERM_LOOKBACK:
                self.allocation_history.pop(0)

            # Store current prices for the lookback calculation
            current_prices = self.dfslice[["coin", "open"]].set_index("coin").to_dict().get("open", {})
            self.price_history.append(current_prices)
            if len(self.price_history) > config.LONG_TERM_LOOKBACK:
                self.price_history.pop(0)
        
        # ✅ REWARD CALCULATION: Choose between Simple Return or Net Return (Profit minus Costs)
        if self.reward_type == "STRUCTURED_CREDIT":
            # --- Structured Credit Assignment Reward ---
            
            # 💡 FIX: Use a dense reward for the initial "cold start" period.
            # Giving zero reward for the first N steps is too sparse and can kill learning.
            if len(self.price_history) < config.LONG_TERM_LOOKBACK:
                # Fallback to simple, dense reward until history is full
                if self.previous_value > 1e-9:
                    raw_reward = step_return + self.long_term_bonus # Include bonus if active
                    self.step_reward = np.clip(raw_reward, -1.0, 1.0)
                    print(f"⏳ Cold Start Reward (step {self.index}/{config.LONG_TERM_LOOKBACK}): Clipped Gross Return {self.step_reward:.4f}")
                else:
                    self.step_reward = 0.0
            else:
                # Once history is full, switch to the structured reward
                
                # 1. Get prices N steps ago and current prices
                prices_then = self.price_history[0] 
                prices_now = self.price_history[-1]

                # 2. Get average allocation over the window
                allocation_window = np.array(self.allocation_history)
                avg_allocations = np.mean(allocation_window, axis=0)

                # 3. Calculate per-asset rewards
                total_structured_reward = 0.0
                risky_asset_returns = []
                
                for i, coin in enumerate(self.episode_coins):
                    asset_idx = i + 1 # 0 is cash
                    
                    price_then = prices_then.get(coin)
                    price_now = prices_now.get(coin)

                    if price_then is not None and price_now is not None and price_then > 1e-9:
                        asset_return = (price_now - price_then) / price_then
                        
                        # ✅ STABILITY FIX: Squash the volatile 30-step return with tanh
                        # This preserves the direction but clips the magnitude to prevent noise.
                        scaled_asset_return = np.tanh(config.STRUCTURED_REWARD_SCALING_FACTOR * asset_return)
                        risky_asset_returns.append(scaled_asset_return) # Use scaled return for market avg
                        
                        avg_alloc = avg_allocations[asset_idx]
                        
                        reward_asset = avg_alloc * scaled_asset_return
                        total_structured_reward += reward_asset
                        
                        if abs(reward_asset) > 1e-5:
                            print(f"💎 Reward {coin}: AvgAlloc {avg_alloc:.2f} * Tanh({asset_return:.2%}) = {reward_asset:.4f}")

                # 4. Calculate reward for cash
                avg_alloc_cash = avg_allocations[0]
                if risky_asset_returns:
                    avg_market_return = np.mean(risky_asset_returns)
                    # reward_cash = avg_alloc_cash * (-1 * avg_market_return)
                    reward_cash = avg_alloc_cash * max(-avg_market_return, 0.0)
                    total_structured_reward += reward_cash
                    
                    if abs(reward_cash) > 1e-5:
                        print(f"💵 Reward Cash: AvgAlloc {avg_alloc_cash:.2f} * AvgMktReturn {-1*avg_market_return:.2%} = {reward_cash:.4f}")

                raw_reward = total_structured_reward + self.long_term_bonus # ✅ FIX: Add long-term bonus
                
                # --- STABLE REWARD: Penalize for Downside Risk ---
                # Instead of unstable division, we subtract a penalty proportional to the downside risk.
                # This encourages stability without causing reward explosion.
                # recent_returns = self.return_history[-WINDOW:]
                downside_returns = [r for r in self.return_history if r < 0]
                if len(downside_returns) > 1:
                    downside_deviation = np.std(downside_returns)
                else:
                    # No downside volatility, no penalty.
                    downside_deviation = 0.0
                
                # Subtract the risk penalty from the skill-based reward
                risk_penalty = config.VOLATILITY_PENALTY_WEIGHT * downside_deviation
                penalized_reward = raw_reward - risk_penalty

                # As requested, clip the final reward to prevent Q-value explosion
                self.step_reward = np.clip(penalized_reward, -1.0, 1.0)
                
                if abs(raw_reward) > 1e-5 or risk_penalty > 1e-5:
                    print(f"🛠️ Structured Reward: Raw {raw_reward:.4f}, RiskPenalty: {risk_penalty:.6f}, Penalized: {penalized_reward:.4f}, Clipped: {self.step_reward:.4f}")

        elif self.previous_value > 1e-9:  # Avoid division by zero
            # Calculate gross return (percentage change in portfolio value)
            gross_return = step_return
            
            if self.reward_type == "TRANSACTION_COST":
                # Net Return = Gross Return - Transaction Costs (as percentage)
                # ✅ CORRECTED LOGIC: Use the correctly calculated transaction_cost
                transaction_cost_pct = (transaction_cost / self.previous_value if self.previous_value > 1e-9 else 0.0)
                
                # --- UPDATE: Include long-term bonus in final reward calculation ---
                raw_reward = gross_return + self.long_term_bonus #- transaction_cost_pct - volatility_penalty
                
                # ✅ REWARD CLIPPING FIX: Clip rewards to prevent Q-value explosion
                # Portfolio percentage returns can occasionally spike during market events
                # Clip to reasonable range to prevent sudden Q-value jumps
                self.step_reward = np.clip(raw_reward, -1.0, 1.0)  # Clip to ±10% per step max
                
                # Track total transaction costs for episode summary
                self.total_transaction_costs += transaction_cost
                
                if transaction_cost > 0 or volatility_penalty > 1e-5 or abs(self.long_term_bonus) > 1e-5:
                    print(f"📊 Gross: {gross_return:.4f}, Tx: {transaction_cost_pct:.4f}, "
                          f"Vol: {volatility_penalty:.4f}, LT: {self.long_term_bonus:.4f}, "
                          f"Raw: {raw_reward:.4f}, Clipped: {self.step_reward:.4f}")
            else:
                # Simple Return (original calculation) + long-term bonus
                raw_reward = gross_return + self.long_term_bonus
                
                # ✅ REWARD CLIPPING FIX: Clip rewards to prevent Q-value explosion
                self.step_reward = np.clip(raw_reward, -1.0, 1.0)  # Clip to ±10% per step max
                
                if abs(self.long_term_bonus) > 1e-5:
                    print(f"📊 Simple Return: {gross_return:.4f} + LT Bonus: {self.long_term_bonus:.4f} = "
                          f"Raw: {raw_reward:.4f}, Clipped: {self.step_reward:.4f}")
        else:
            # Only long-term bonus if no value change
            raw_reward = self.long_term_bonus
            self.step_reward = np.clip(raw_reward, -1.0, 1.0)  # Clip to ±10% per step max

    def get_observations(self):
        """Get scaled observations from current market data"""
        dfslice = self.dfslice
        dfs = pd.DataFrame()
        for i, grp in dfslice.groupby("coin"):
            tempdf = pd.DataFrame(self.scaler.transform(grp[config.SCOLS].values))
            tempdf.columns = [i+"_"+c for c in config.SCOLS]
            if dfs.empty:
                dfs = tempdf
            else:
                dfs = dfs.merge(tempdf, right_index=True, left_index=True, how='inner')

        return dfs

    def get_observations_unscaled(self):
        """Get unscaled observations from current market data"""
        dfslice = self.dfslice
        dfs = pd.DataFrame()
        for i, grp in dfslice.groupby("coin"):
            tempdf = pd.DataFrame(grp[config.COLS].values)
            tempdf.columns = [i+"_"+c for c in config.COLS]
            if dfs.empty:
                dfs = tempdf
            else:
                dfs = dfs.merge(tempdf, right_index=True, left_index=True, how='inner')
        
        self.memory_return = pd.concat([self.memory_return, dfs[[t+"_close" for t in self.episode_coins]]], ignore_index=True)
        
        return dfs

    def calculate_actual_shares_from_money_split(self):
        """Calculate number of shares to buy based on money allocation"""
        dict_coin_price = self.dfslice[["coin", "open"]].set_index("coin").to_dict()["open"]
        
        # --- FIX: Directly use the agent's target cash allocation ---
        # This is robust to floating point errors and avoids calculating cash as a
        # remainder, which was the source of the negative cash values.
        self.current_cash = self.money_split_ratio[0] * self.current_value
        
        # --- NEW: Store the agent's target allocation before it's altered by market drift ---
        self.last_target_allocation = self.money_split_ratio.copy()
        
        num_shares = []
        for i, c in enumerate(self.episode_coins):
            money_for_coin = self.money_split_ratio[i+1] * self.current_value
            
            if c in dict_coin_price and dict_coin_price[c] > 1e-9:
                num_shares.append(money_for_coin / dict_coin_price[c])
            elif c in self.old_dict_coin_price_1 and self.old_dict_coin_price_1[c] > 1e-9:
                num_shares.append(money_for_coin / self.old_dict_coin_price_1[c])
            else:
                num_shares.append(0.0) # Cannot buy if price is unknown or zero
            
        for c in dict_coin_price:
            self.old_dict_coin_price_1[c] = dict_coin_price[c]
        
        return num_shares

    def calculate_money_from_num_stocks(self):
        """Calculate current portfolio value from number of shares"""
        money_dist = []
        # Cash is the first element, and it doesn't change with market prices
        money_dist.append(self.current_cash)
        dict_coin_price = self.dfslice[["coin", "open"]].set_index("coin").to_dict()["open"]
        
        for i, c in enumerate(self.episode_coins):
            price = dict_coin_price.get(c, self.old_dict_coin_price_2.get(c, 0))
            money_dist.append(self.current_stock_num_distribution[i] * price)
            self.old_dict_coin_price_2[c] = price # Update last known price
            
        return money_dist, sum(money_dist)

    def normalize_money_dist(self):
        """Normalize money distribution to get portfolio weights"""
        if self.use_variable_portfolio:
            # For variable portfolios, we need to return a fixed-size numpy array
            # padded with zeros for inactive coins.
            normal = np.zeros(config.MAX_COINS + 1)
            if self.current_value > 1e-9:
                # Calculate weights for cash and active coins
                for i, c_val in enumerate(self.current_stock_money_distribution):
                    normal[i] = c_val / self.current_value
            else:
                normal[0] = 1.0 # Default to all cash if value is zero
            return normal
        else:
            # For fixed portfolios, a simple list is sufficient
            normal = []
            if self.current_value > 1e-9:
                for i, c in enumerate(self.current_stock_money_distribution):
                    normal.append(c / self.current_value)
            else:
                normal = [0.0] * len(self.current_stock_money_distribution)
                if len(normal) > 0:
                    normal[0] = 1.0
            return normal

    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Time: {self.current_time}")
            print(f"Portfolio Value: ${self.current_value:.2f}")
            print(f"Cash Ratio: {self.money_split_ratio[0]:.3f}")
            
            if self.use_variable_portfolio:
                print(f"Active Coins: {self.episode_coins}")
                for i, coin in enumerate(self.episode_coins):
                    print(f"{coin} Ratio: {self.money_split_ratio[i+1]:.3f}")
            else:
                for i, coin in enumerate(config.COINS):
                    print(f"{coin} Ratio: {self.money_split_ratio[i+1]:.3f}")
                    
            print(f"Step Reward: {self.step_reward:.4f}")
            print("-" * 40)

    def close(self):
        """Clean up environment resources"""
        pass