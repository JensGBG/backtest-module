import numpy as np
import pandas as pd
import talib as ta
import yfinance as yf
import time
import itertools

from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot as plt

# TODO:
# 1.) Implement comission logic.
# 2.) Implement Walk-Forward permutation test (very computational heavy!).
# 3.) Implement shifting logic when wanting to enter the same day as signal.
# 4.) Improve partitioning of walk-forward analysis.
# 5.) Double check permutation logic of ohlc.


def moving_average_signal(data, *parameters):
    '''
    How the signal should be created:
    Input: Only parameters nothing else.
    Output: A np.ndarray of shape = (N, A) <--- VERY IMPORTANT!!!
        N: The number of bars in our original df.
        A: The number of tradable assets.
    Values must strictly be: (1, 0, -1) corresponding to (Long position, no position, Short position).
    '''
    ######## ENTER STRATEGY HERE ########
    close = data[:,0]
    ma_short = ta.SMA(close, timeperiod=parameters[0])
    ma_long = ta.SMA(close, timeperiod=parameters[1])

    signal = np.where(
        (ma_short > ma_long) &
        (close > ma_short) &
        (close > ma_long),
        1, 0
        )
    signal = np.where(
        (ma_short < ma_long),
        0, signal
        )
    
    signal_matrix = np.array([signal]).T

    #### float test
    signal_matrix = signal_matrix

    return signal_matrix # This object should be NxA np.ndarray, where N is number of days, and A is number of tradable assets.
    # Each column
    #####################################


class Backtest:
    def __init__(
            self,
            df: pd.DataFrame,
            signal_function,
            asset_columns: list[str],
            signal_columns: list[str],
            ohlc_columns: list[str] = None,
            non_financial_columns: list[str] = None,
            comission=0.005,
            precentage_comission=True):
        '''
        Inputs:
            df: pd.Dataframe with all data needed.
            signal_function: The function used for creating a signal.
            columns_assets: List of strings of columns we will trade on.
            columns_signal: List of strings of columns passed to signal_function.
            ohlc_columns: List of lists where the inner lists are name of columns which are of the type OHLC.
                Example:
                [['Open stock A', 'High stock A', 'Low stock A', 'Close stock A'],
                 ['Open stock B', 'High stock B', 'Low stock B', 'Close stock B']]
                Observe that column names of inner list needs to be in the order: OHLC.
                If None given, it is assumed that each column is part of different price data.
            non_financial_columns: These columns will be permuted using normal permutation as supposed to permutating bar to bar price changes.
            comission: If precentage_comission==True: Precentage cost of each trade at entry and exit. Otherwise it will be a fixed sum at each entry and exit.
        '''
        # Asserting all columns given in non_financial_columns exists.
        if non_financial_columns != None:
            assert set(non_financial_columns).issubset(set(df.columns)), f"Missing columns: {set(non_financial_columns) - set(df.columns)}"

        # Asserting all columns given in ohlc_columns exists.
        if ohlc_columns != None:
            for cols in ohlc_columns:
                assert set(cols).issubset(set(df.columns)), f"Missing columns: {set(cols) - set(df.columns)}"

        self.signal_function = signal_function

        # Create a mapping from column name to index
        all_column_indices = np.arange(len(df.columns))
        col_to_idx = {col: idx for idx, col in enumerate(df.columns)}

        taken_indices = []
        # Storing indices of non-financial columns in self.non_financial_indices.
        if non_financial_columns != None:
            self.non_financial_indices = [col_to_idx[name] for name in non_financial_columns]
            taken_indices = taken_indices + self.non_financial_indices
        else:
            self.non_financial_indices = None


        # Convert jagged array of names -> jagged array of indices
        if ohlc_columns != None:
            self.ohlc_indices = [[col_to_idx[name] for name in sublist] for sublist in ohlc_columns]
            for indices in self.ohlc_indices:
                for idx in indices:
                    taken_indices.append(idx)
        else:
            self.ohlc_indices = None
        
        self.signal_indices = [col_to_idx[name] for name in signal_columns]
        self.asset_indices = [col_to_idx[name] for name in asset_columns]


        normal_indices = []
        for index in all_column_indices:
            if index not in taken_indices:
                normal_indices.append(index)
        self.normal_indices = normal_indices

        data_np = df.to_numpy()
        ### Scaling OHLC data so that first Open price begins at 100.0 ###
        if self.ohlc_indices != None:
            for cols in self.ohlc_indices:
                if len(cols) != 0:
                    data_np[:, cols] = 100 * (data_np[:, cols] / data_np[0, cols][0])
        
        ### Scaling normal financial data so that first value begins at 100.0 ###
        data_np[:, self.normal_indices] = 100 * (data_np[:, self.normal_indices] / data_np[0, self.normal_indices])

        self.data_np = data_np
        df[:] = data_np
        self.df = df

        # TODO:##
        self.comission = comission
        self.precentage_comission = precentage_comission # Is comission given as a fixed amount our a precentage?
        ######


        self.asset_data_np = self.data_np[:, [col_to_idx[col] for col in asset_columns]] # Copy columns of assets we want to trade to numpy matrix.
        self.signal_data_np = self.data_np[:, [col_to_idx[col] for col in signal_columns]] # Copy columns of signals we want to trade on to numpy matrix.

    ### Metrics ###
    def get_metric(self, equity_matrix, metric=None):
        for key in self.metric_map:
            if key==metric:
                return self.metric_map[key](equity_matrix)
        print("Valid metrics are: ")
        for key in self.metric_map:
            print(f"{key} ")
        raise TypeError("Please enter a valid metric.")
    
    @staticmethod
    def profit_factor(equity_matrix):
        gross_diff = np.diff(equity_matrix, axis=0)

        gross_win = gross_diff[gross_diff > 0].sum()
        gross_loss = gross_diff[gross_diff < 0].sum()

        if float(gross_loss)==0.0:
            if float(gross_win)==0.0:
                return 1
            gross_loss = - gross_win

        profit_factor = -gross_win/gross_loss

        return profit_factor
    

    metric_map = {
        "profit-factor" : profit_factor
    }

    # Create entry matrix and shift one bar down (entry_matrix will lag with one bar).
    
    @staticmethod
    def compute_backtest(signal_matrix, asset_matrix, return_each_position_column=False):
        # Translates signal_matrix into weighted signals
        # signal_matrix_weights = np.abs(signal_matrix)
        # signal_matrix_row_weights = signal_matrix_weights.sum(axis=1, keepdims=True)
        # signal_matrix_row_weights[signal_matrix_row_weights==0] = 1.0 # Handling division by zero.
        # signal_matrix = signal_matrix / signal_matrix_row_weights

        entry_matrix = np.zeros_like(signal_matrix, dtype='float64') # In this matrix we will store our entries. (1, 0, -1) = (long, no position, short)
        entry_matrix[2:] = signal_matrix[:-2] # This shifts the entry matrix down by 2 bars. 1 bar for accounting for lag (we can enter a position first after signal),
        # and another bar for computation of profit and losses.
        


        asset_matrix_log_diff = np.zeros_like(signal_matrix, dtype='float64') # In this matrix we will store the differences of log prices.
        asset_matrix_log = np.log(asset_matrix) # Log all prices so we can easily compute profit-loss by addition and subtraction.

        asset_matrix_log_diff_temp = np.diff(asset_matrix_log, axis=0) # asset_matrix_log_diff_temp will be one row shorter!
        asset_matrix_log_diff[1:] = asset_matrix_log_diff_temp # Puts in the log-difference, observe that last row will be zeros. EDIT First row will be zeros!!!
        
        equity_matrix_log_diff = asset_matrix_log_diff * entry_matrix

        equity_matrix_log_diff[0] = np.log(asset_matrix[0]) # Sets the first row to the initial value of our assets.

        equity_matrix_log = np.cumsum(equity_matrix_log_diff, axis=0) # Adding up all the wins and losses.
        equity_matrix = np.exp(equity_matrix_log) # Exponentiate all prices to get back to normal prices.
        
        if return_each_position_column==False: # Returns the total value of the given portfolio.
            equity_matrix_mean = np.mean(equity_matrix, axis=1)
            equity_matrix_mean = np.array([equity_matrix_mean]).T
            return equity_matrix_mean
        else:
            return equity_matrix # Returns a np.ndarray where each column is the value of a position.

    def run_backtest(self, *parameters, return_each_position_column=False): # , take_pos_one_bar_after_signal=True
        signal_matrix = self.signal_function(self.signal_data_np, *parameters)
        asset_matrix = self.asset_data_np

        return self.compute_backtest(signal_matrix=signal_matrix, asset_matrix=asset_matrix, return_each_position_column=return_each_position_column)
    

    def run_backtest_on_given_data(self, asset_data, signal_data, *parameters, return_each_position_column=False): # , take_pos_one_bar_after_signal=True
        signal_matrix = self.signal_function(signal_data, *parameters)

        return self.compute_backtest(asset_matrix=asset_data, signal_matrix=signal_matrix, return_each_position_column=return_each_position_column)

    def find_best_param(self, *parameter_vectors, metric=list(metric_map)[0]):
        '''
        Given parameter vectors, it finds the best parameter combination for the original data given.
        Returns: (best_param_values, best_metric, best_equity_matrix)
        '''
        parameter_combinations = list(itertools.product(*parameter_vectors))

        # Creating dictionaries storing equity curves and metrics for each param comb.
        param_equity_dict = {}
        param_metric_dict = {}
        for params in parameter_combinations:
            param_equity_dict[params] = self.run_backtest_on_given_data(self.asset_data_np, self.signal_data_np, *params)
            param_metric_dict[params] = self.get_metric(equity_matrix=param_equity_dict[params], metric=metric)


        best_param_values, best_metric = max(param_metric_dict.items(), key=lambda item: item[1])
        best_equity_matrix = param_equity_dict[best_param_values]

        return best_param_values, best_metric, best_equity_matrix

    def find_best_param_for_data(self, asset_data, signal_data, *parameter_vectors, metric=list(metric_map)[0]):
        '''
        Given parameter vectors, it finds the best parameter combination for the data given (usually a permutation).
        Returns: (best_param_values, best_metric, best_equity_matrix)
        '''
        parameter_combinations = list(itertools.product(*parameter_vectors))

        equity_matrices = [self.run_backtest_on_given_data(asset_data, signal_data, *params, return_each_position_column=False) for params in parameter_combinations]
        metric_array = [self.get_metric(equity_matrix=equity_matrix, metric=metric) for equity_matrix in equity_matrices]

        best_metric = np.max(metric_array)
        best_metric_idx = np.argmax(metric_array)
        best_param_values = parameter_combinations[best_metric_idx]
        best_equity_matrix = equity_matrices[best_metric_idx]

        return best_param_values, best_metric, best_equity_matrix

    def permute_ohlc_data(self, ohlc, perm_idx):
        N = len(ohlc)
        O, H, L, C = ohlc.T

        r_cc = C[1:] / C[:-1]
        o_rel = O[1:] / C[1:]
        h_rel = H[1:] / C[1:]
        l_rel = L[1:] / C[1:]

        bar_units = np.column_stack([r_cc, o_rel, h_rel, l_rel])
        shuffled = bar_units[perm_idx]

        new_O = np.empty(N)
        new_H = np.empty(N)
        new_L = np.empty(N)
        new_C = np.empty(N)

        new_O[0], new_H[0], new_L[0], new_C[0] = O[0], H[0], L[0], C[0]

        for i in range(1, N):
            r, o_r, h_r, l_r = shuffled[i-1]
            new_C[i] = new_C[i-1] * r
            new_O[i] = o_r * new_C[i]
            new_H[i] = h_r * new_C[i]
            new_L[i] = l_r * new_C[i]

        return np.column_stack([new_O, new_H, new_L, new_C])



    @staticmethod
    def permutate_normal_data(data: np.array, perm_idx):
        """
        Takes in a numpy array and returns a numpy array.
        Data needs to be handled prior and post this function, but the function itself is fast.
        """
        if np.any(data.astype(float) == 0.0):
            raise ValueError("Data contains zero values, which can lead to division by zero errors during log transformation. \n" \
            "Please ensure that non_financial_columns are specified.")
        data_log = np.log(data)
        data_log_diff = np.diff(data_log, axis=0) # OBS! This array is one element shorter than the original data_log
        
        #data_log_diff_perm = np.random.permutation(data_log_diff)
        data_log_diff_perm = data_log_diff[perm_idx]
        data_log_diff_perm = np.concatenate((np.array([data_log[0]]), data_log_diff_perm), axis=0)
        data_log_diff_perm = np.cumsum(data_log_diff_perm, axis=0)
        data_log_diff_perm = np.exp(data_log_diff_perm)  # Convert back to price scale

        return data_log_diff_perm

    def create_permutation(self, N=10):
        """
        Takes in a numpy array and returns a numpy array.
        Data needs to be handled prior and post this function, but the function itself is fast.
        """
        data_permutations = np.empty(shape=(N, self.data_np.shape[0], self.data_np.shape[1])) # Initialising empty array.

        for n in range(N):
            perm_idx = np.random.permutation(self.data_np.shape[0] - 1) # Every permutation method will share permutation seed, meaning each column
                                                                            # will be permutated in the same order.

            # Iterating through each set of OHLC data and permutating.
            if self.ohlc_indices != None:
                for column_indices in self.ohlc_indices:
                    if len(column_indices)!=0:
                        ohlc_perm = self.permute_ohlc_data(ohlc=self.data_np[:, column_indices], perm_idx=perm_idx)
                        data_permutations[n, :, column_indices] = ohlc_perm.T


            ### Permutating "normal financial data". ###
            data_permutations[n, :, self.normal_indices] = self.permutate_normal_data(self.data_np[:, self.normal_indices], perm_idx=perm_idx).T


            ### Permutating "non financial data". ###
            if self.non_financial_indices != None:
                data_to_be_perm_non_financial = self.data_np[:, self.non_financial_indices]
                data_perm = data_to_be_perm_non_financial
                data_perm[1:] = data_to_be_perm_non_financial[perm_idx]
                data_permutations[n, :, self.non_financial_indices] = data_perm.T

        return data_permutations

    def run_perm_test(self, *parameter_vectors, N=10):
        # Create permutations.
        data_permutations = self.create_permutation(N=N)

        best_params_array = np.empty(shape=(N, len(parameter_vectors)))
        best_metric_array = np.empty(N)
        best_equity_matrix_array = np.empty(shape=(N, self.asset_data_np.shape[0], self.asset_data_np.shape[1]))

        # for each permutation, backtest on all parameter combination, return best ones.
        for n, data_perm in enumerate(data_permutations):
            signal_perm = data_perm[:, self.signal_indices]
            asset_perm = data_perm[:, self.asset_indices]
            best_params_array[n] , best_metric_array[n], best_equity_matrix_array[n] = self.find_best_param_for_data(asset_perm, signal_perm, *parameter_vectors, metric="profit-factor")


        self.best_params_array = best_params_array
        self.best_metric_array = best_metric_array
        self.best_equity_matrix_array = best_equity_matrix_array
        self.number_of_perms = N
        return best_params_array, best_metric_array, best_equity_matrix_array
    

        # We have now asserted total length for each period.
    def partition_walk_forward_data(self, n_splits=8, train_size=300, test_size=100, plot=True):
        # number_of_bars = self.data_np.shape[0]
        # length_of_iteration = number_of_bars / (2*in_sample_precentage - n_splits*in_sample_precentage + n_splits - 1)
        # train_size = round(in_sample_precentage * length_of_iteration)
        # test_size = round(length_of_iteration - train_size)
        tscv = TimeSeriesSplit(n_splits=n_splits, 
                            test_size=test_size,
                            max_train_size=train_size
                            )
        train_splits, test_splits = [], []
        for train_idx, test_idx in tscv.split(np.arange(self.data_np.shape[0])):
            train_splits.append(train_idx)
            test_splits.append(test_idx)

        self.train_splits = train_splits
        self.test_splits = test_splits

        if plot:
            fig, ax = plt.subplots(figsize=(12, 4))

            bar_height = 1.0
            for i, (train_idx, test_idx) in enumerate(zip(train_splits, test_splits)):
                if len(train_idx) > 0:
                    ax.broken_barh(
                        [(train_idx[0], len(train_idx))],
                        (i, bar_height),
                        facecolors='blue'
                    )
                ax.broken_barh(
                    [(test_idx[0], len(test_idx))],
                    (i, bar_height),
                    facecolors='orange'
                )

            ax.set_yticks(range(len(train_splits)))
            ax.set_yticklabels([f"Split {i+1}" for i in range(len(train_splits))])
            ax.set_xlabel("Time index")
            ax.set_ylabel("Splits")

            # Vertical grid for time steps
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            # Horizontal grid between splits
            ax.set_yticks(np.arange(0, len(train_splits)+1, 1), minor=True)  # boundaries
            ax.grid(axis="y", linestyle="--", linewidth=1, alpha=0.5)

            plt.tight_layout()
            plt.show()
        return None


    def run_walk_forward(self, *parameter_vectors, metric=list(metric_map)[0], return_each_position_column=False):
        wf_signal_matrix = np.zeros(shape=self.signal_data_np.shape)
        best_params_list = []


        # Finding best parameters for each training set.
        for i, train_split in enumerate(self.train_splits):
            best_param_values, best_metric, best_equity_matrix = self.find_best_param_for_data(self.asset_data_np[train_split], self.signal_data_np[train_split], *parameter_vectors, metric=metric)
            best_params_list.append(best_param_values)

        self.wf_params = best_params_list

        # Creating signals for each test set.
        for i, test_split in enumerate(self.test_splits):
            params = best_params_list[i]
            signals = self.signal_function(self.signal_data_np, *params)
            wf_signal_matrix[test_split] = signals[test_split]
        
        # Running backtest given signals.
        self.wf_equity = self.compute_backtest(signal_matrix=wf_signal_matrix, asset_matrix=self.asset_data_np, return_each_position_column=return_each_position_column)
        return self.wf_equity

    def plot_permutation_result(self, optimal_profit_factor, bins=20):
        plt.hist(self.best_metric_array, bins=bins, color='lightblue', label="Optimized permutated portfolios")
        plt.title(f"Distribution of Profit Factors, $N={self.number_of_perms}$")
        plt.axvline(optimal_profit_factor, color='blue', label="Real portfolio")
        plt.legend()
        plt.show()

        p_level = np.sum(optimal_profit_factor > self.best_metric_array) / len(self.best_metric_array)
        print(f"p-level after permutation test: {round(1 - p_level, 5)*100}%")
        return None

    def plot_walk_forward(self):
        # Plotting equity curve of walk-forward optimization.
        plt.plot(self.df.index, self.wf_equity[:,0], label="Walk-Forward equity")
        plt.plot(self.df.index, np.mean(self.asset_data_np, axis=1), label="Buy-n-Hold equity")
        plt.title("Walk forward optimization performance")
        plt.grid()
        plt.legend()
        plt.show()

        number_of_parameters = len(self.wf_params[0])
        # Unpacking parameter values to lists.
        param_matrix = np.zeros(shape=(len(self.wf_params), number_of_parameters))
        for n in range(number_of_parameters):
            for p_idx in range(len(self.wf_params)):
                param_matrix[p_idx, n] = self.wf_params[p_idx][n]

        dates_array = []
        for train_split in self.train_splits:
            idx = train_split[-1]
            dates_array.append(self.df.index[idx])
        
        fig, axes = plt.subplots(nrows = number_of_parameters, ncols=1, figsize=(8, 2*number_of_parameters), sharex=True)

        for i, ax in enumerate(axes):
            ax.plot(dates_array, param_matrix[:, i], marker='o')
            ax.grid()
        
        plt.suptitle("Optimal parameters through out time")
        plt.show()

        return None




if __name__ == "__main__":
    start_date, end_date = "2016-01-01", "2022-01-01"
    ticker = "INVE-B.ST"
    interval = '1d'
    df = yf.download(tickers=ticker, start=start_date, end=end_date, multi_level_index=False, interval=interval)
    


    backtest = Backtest(df=df,
                        asset_columns=['Close'],
                        signal_columns=['Close'],
                        signal_function=moving_average_signal,
                        #ohlc_columns=[['Open', 'High', 'Low', 'Close']],
                        non_financial_columns=['Volume'],
                        comission=0.0,
                        precentage_comission=True
                            )
    param_vector_1 = np.arange(20, 70, 4)
    param_vector_2 = np.arange(120, 240, 4)

    action = 'permutation test'
    print(backtest.df)

    if action=='walk-forward test':
        backtest.partition_walk_forward_data(n_splits=11, train_size=300, test_size=100, plot=True)
        wf_backtest = backtest.run_walk_forward(param_vector_1, param_vector_2, metric="profit-factor")
        backtest.plot_walk_forward()

    if action=='permutation test':

        metric = "profit-factor"
        buy_n_hold_profit_factor = backtest.get_metric(df.to_numpy(), metric)
        optimal_params, optimal_profit_factor, optimal_equity_array = backtest.find_best_param(param_vector_1, param_vector_2, metric=metric)

        df = pd.read_excel('backtest_results.xlsx', sheet_name="general-metrics", index_col=0)
        df[f"{ticker}-{start_date}-{end_date}"] = [optimal_profit_factor, buy_n_hold_profit_factor]
        with pd.ExcelWriter("backtest_results.xlsx", engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name="general-metrics")


        N = 1000

        start_time = time.perf_counter()
        best_params_array, best_metric_array, best_equity_matrix_array = backtest.run_perm_test(param_vector_1, param_vector_2, N=N)
        end_time = time.perf_counter()
        print(f"Time it took for N = {N}: {end_time - start_time}")

        plt.hist(best_metric_array, bins=20, color='orange')
        plt.title(f"Distribution of Profit Factors, $N={N}$")
        plt.axvline(optimal_profit_factor, color='blue', label="Real portfolio")
        plt.show()

        p_level = np.sum(optimal_profit_factor > best_metric_array) / len(best_metric_array)
        print(f"p-level after permutation test: {round(1 - p_level, 5)*100}%")