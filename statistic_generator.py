import numpy as np
import pandas as pd
# from load_symbol import LoadYahooSymbol
from line_printer import LinePrinter
import collections


class StatisticGenerator:
    def __init__(self, open_column_name, high_column_name, low_column_name, close_column_name, volume_column_name,
                 min_data_volatility, sentence_length, future_data_length, max_date_gap, default_column_name,
                min_number_of_sentences=50, rounding_precision=4):
        """
        This class provides utility functions to evaluate the data.
        These functions include:
        1. check_usability_of_data
        2. Max single bar movement
        3. Calculate statistics about a given symbol
            The statistics is calculated on 3 separate parts of the dataset. The number 3 was chosen because of convininience
            Basically the data is split into 3 separate equal parts and the following statistics is provided on each part and
            also on the entire dataset. Volatility is calculated as (max-min)/min in each slice
            The statistics are as follows:
            3.1: volatility_accepted_part_1-3 and total : if the volatility in each part is above the minimum set volatility
            3.2: chunks_volatility_part_1-3 and total: the actual volatilty in that part
            3.3: draw_down_part_1-3 and total: actual drawdown in each part
            3.4: max_drop_single_bar_part_1-3 and total: maximum drop in a single bar
            3.5: max_gain_single_bar_part_1-3 and total: maximum gain in a single bar
            3.6: mean_red_pct_part_1-3 and total: mean value of all the red candles. red defined as close<open
            3.7: mean_green_pct_part_1-3 and total: mean value of all the green candles. green defined as close>open
            3.8: first_date: first date of available data
            3.9: last_date: last date of available data
            3.10: total_return: total return from inception (first candle open-last candle close)/first candle open
            3.11: total_no_of_rows of available data

        :param open_column_name: the name of column which represent open
        :param close_column_name: the name of column which represent close
        :param min_data_volatility: the minimum volatility for which data should be categorized as acceptable.
        If price moves < min_data_volatility in any part or in the entire dataset, then symbol is not good for trading
        :param sentence_length: is the length of data that will be used for generating data for the learning agent
        :param future_data_length: how long into the future will the system try to predict. These two parameters
        help determine how many data points can be generated from any symbol
        :param max_date_gap: time delta for the maximum data gap acceptable for the data
        :param default_column_name: this is the default column name used throughout this class.
        In most cases, this should be set to close
        :param number_of_chunks: This is the number of parts the data should be split in order to check for volatility
        and to see if the data is usable for training purposes
        :param min_number_of_sentences: minimum number of sentences to consider that the symbol has enough data
        :param rounding_precision: precision level
        """
        self.rounding_precision = rounding_precision
        self.open_column_name = open_column_name
        self.close_column_name = close_column_name
        self.high_column_name = high_column_name
        self.low_column_name = low_column_name
        self.volume_column_name = volume_column_name
        self.min_data_volatility = min_data_volatility
        self.sentence_length = sentence_length
        self.future_data_length = future_data_length
        self.total_row_length_for_learning_agent = self.sentence_length + self.future_data_length
        self.max_date_gap = max_date_gap
        self.min_number_of_sentences = min_number_of_sentences
        # self.number_of_chunks = number_of_chunks
        self.default_column_name = default_column_name
        self._line_printer = LinePrinter()

    def check_integrity_of_data(self, data_to_check_integrity, file_path, file_name):
        """
        make sure to use parse_dates=True when initially loading data

        This function checks the very basics of the data to make sure it is usable for further processing.
        These checks include:
        0. Make sure there are more than self.total_row_length_for_learning_agent data available in the symbol
        1. Make sure all data values are positive [except volume]
        2. if Volume data is available, it will check the % and number of volume data above 0
        3. compares open - close and high - low and calculates the percentage of data where they are equal
        4. make sure that the index of the data is a date object
        5. make sure that there are no gaps in the data that are larger than self.max_date_gap
        # todo: it is still possible to use the data even if there is a gap. We must be able to check if
        # even though there is a gap, does the data still have parts that are usable

        :param file_name:
        :param file_path:
        :param data_to_check_integrity: the complete dataset
        :return: dic{'has_enough_data': bool,
                    'all_data_values_are_positive': bool,
                    'percent_of_vol_data_equal_to_0': float,
                    'number_of_vol_data_equal_to_0': int,
                    'percent_of_low_equal_to_high': float,
                    'percent_of_open_equal_to_close': float,
                    'index_is_date_time_object': bool,
                    'no_gaps_larger_than_'+self.max_date_gap: bool
                    }
        """
        integrity_check_results = {'file_name': file_name,
                                   'file_path': file_path,
                                   'number_of_rows': len(data_to_check_integrity),
                                   'has_enough_data': bool,
                                   'sentence_length': self.sentence_length,
                                   'look_into_future': self.future_data_length,
                                   'number_of_available_sentences': int,
                                   'all_data_values_are_positive': bool,
                                   'percent_of_vol_data_equal_to_0': float,
                                   'number_of_vol_data_equal_to_0': int,
                                   'percent_of_low_equal_to_high': float,
                                   'percent_of_open_equal_to_close': float,
                                   'index_is_date_time_object': bool,
                                   'max_time_gap': int
                                   }

        # Checking length of available data

        total_number_of_sentences = (len(data_to_check_integrity) - self.sentence_length + 1) - self.future_data_length
        if total_number_of_sentences > self.min_number_of_sentences:
            integrity_check_results['has_enough_data'] = True
        else:
            integrity_check_results['has_enough_data'] = False

        integrity_check_results['number_of_available_sentences'] = total_number_of_sentences
        # Checking positive data
        check_open = np.sign(data_to_check_integrity[self.open_column_name]) < 0
        check_high = np.sign(data_to_check_integrity[self.high_column_name]) < 0
        check_low = np.sign(data_to_check_integrity[self.low_column_name]) < 0
        check_close = np.sign(data_to_check_integrity[self.close_column_name]) < 0
        check_volume = np.sign(data_to_check_integrity[self.volume_column_name]) < 0

        if (len(data_to_check_integrity[check_open])) | (len(data_to_check_integrity[check_high])) | \
                (len(data_to_check_integrity[check_low])) | (len(data_to_check_integrity[check_close])) | \
                (len(data_to_check_integrity[check_volume])):
            integrity_check_results['all_data_values_are_positive'] = False
        else:
            integrity_check_results['all_data_values_are_positive'] = True

        # 'percent_of_vol_data_equal_to_0':
        # if statement just to avoid division by 0 in an unforeseen case

        if len(data_to_check_integrity[self.volume_column_name]) > 0:
            integrity_check_results['percent_of_vol_data_equal_to_0'] = round(
                len(data_to_check_integrity[data_to_check_integrity[self.volume_column_name] == 0]) / \
                len(data_to_check_integrity[self.volume_column_name]), self.rounding_precision)

        integrity_check_results['number_of_vol_data_equal_to_0'] = len(
            data_to_check_integrity[data_to_check_integrity[self.volume_column_name] == 0])

        # 'percent_of_low_equal_to_high'
        integrity_check_results['percent_of_low_equal_to_high'] = \
            round(len(data_to_check_integrity[data_to_check_integrity[self.high_column_name] ==
                                              data_to_check_integrity[self.low_column_name]]), self.rounding_precision)

        # 'percent_of_open_equal_to_close'
        integrity_check_results['percent_of_open_equal_to_close'] = \
            round(len(data_to_check_integrity[data_to_check_integrity[self.open_column_name] ==
                                              data_to_check_integrity[self.close_column_name]]),
                  self.rounding_precision)

        # 'index_is_date_time_object':

        if isinstance(data_to_check_integrity.index.values[0], np.datetime64):
            integrity_check_results['index_is_date_time_object'] = True
        else:
            integrity_check_results['index_is_date_time_object'] = False

        # time gap calculator
        time_gap_analyzer = pd.DataFrame()

        time_gap_analyzer['tvalue'] = data_to_check_integrity.index
        time_gap_analyzer['delta'] = (time_gap_analyzer['tvalue'] - time_gap_analyzer['tvalue'].shift()). \
            astype('timedelta64[D]').fillna(0)

        integrity_check_results['max_time_gap'] = time_gap_analyzer['delta'].max()

        return integrity_check_results

    def calc_max_dd(self, data_column):
        roll_max = data_column.cummax()

        daily_draw_down = data_column / roll_max - 1.0

        max_daily_draw_down = daily_draw_down.cummin()

        return round(max_daily_draw_down.min(), self.rounding_precision)

    def calculate_max_single_bar_gain_loss(self, entire_data, calculate_using_open_to_close=True,
                                           verbose=False):

        """
        This function calculates the difference between open and close in each bar.

        This function assumes index is a date object

        :param calculate_using_open_to_close: if this is true, then movement is calculated using (close-open)/open
        if False-> (previous self.default_column_name - current self.default_column_name)/previous self.default_column_name
        :param entire_data: must be a data frame with index set as time
        :param verbose:
        """
        single_bar_movements = {'min_movement': float,
                                'max_movement': float,
                                'min_movement_date': None,
                                'max_movement_date': None,
                                'negative_bars_mean': float,
                                'positive_bars_mean': float
                                }
        # todo: can add a function parameter to determine which column name is going to be used for difference calculation

        # if column_name and calculate_using_open_to_close:
        #     raise Warning('You have provided a column name and calculating bar movements using open and close'
        #                   'column names. The column name you provided is redundant and will not be used by the '
        #                   'function')

        # todo: we could add an option to calculate the difference using previous close if close-open = 0

        if calculate_using_open_to_close:
            difference = ((entire_data[self.close_column_name] - entire_data[self.open_column_name]) /
                          entire_data[self.open_column_name])
        else:
            difference = entire_data[self.default_column_name].diff().fillna(0) / \
                         entire_data[self.default_column_name].shift().fillna(1)

        difference.reset_index(drop=True, inplace=True)
        if verbose:
            print(difference)

        single_bar_movements['min_movement'] = round(difference.min(), self.rounding_precision)
        single_bar_movements['max_movement'] = round(difference.max(), self.rounding_precision)

        # if single_bar_movements['min_movement'] == np.inf:

        single_bar_movements['min_movement_date'] = \
            entire_data.iloc[difference[difference == difference.min()].dropna().index].index.values[0]
        single_bar_movements['max_movement_date'] = \
            entire_data.iloc[difference[difference == difference.max()].dropna().index].index.values[0]

        negative_bars = difference[difference < 0]
        positive_bars = difference[difference > 0]

        if verbose:
            print('negative_bars_mean: ', round(negative_bars.mean(), self.rounding_precision))
            print('positive_bars_mean: ', round(positive_bars.mean(), self.rounding_precision))

        single_bar_movements['positive_bars_mean'] = round(positive_bars.mean(), self.rounding_precision)
        single_bar_movements['negative_bars_mean'] = round(negative_bars.mean(), self.rounding_precision)

        return single_bar_movements

    # def check_col_name(self, column_name):
    #     """
    #     This function simply checks to see if column_name is provided by the user.
    #     If not, it assigns the default column name to it
    #     :param column_name:
    #     :return:
    #     """
    #     if not column_name:
    #         return self.default_column_name
    #     else:
    #         return column_name

    def calculate_volatility(self, data_column) -> float:
        """
        This function calculates (max/min)-1
        :param data_column: the column of data to calculate volatility
        :return: float (max/min)-1
        """
        min_ = data_column.min()
        max_ = data_column.max()

        ratio = 1
        if min_ > 0.01:
            ratio = round((max_ / min_) - 1, self.rounding_precision)

        return ratio

    def original_check_usability_of_data(self, entire_data,
                                         calculate_bar_movements_using_open_to_close=True):  # , verbose=False):

        """
        This function assumes that the index for data is a date object
        :param calculate_bar_movements_using_open_to_close: This parameter determines if price movement is calculated
        using close-open or if it is calculated using the following formula:
        (previous self.default_column_name-current self.default_column_name)/previous self.default_column_name
        :param entire_data: the dataframe containing all the information
        :param verbose: if True it will print some statistics from this function

        :return:  usability_dictionary = {'min_movement': float,
                                'max_movement': float,
                                'min_movement_date': None,
                                'max_movement_date': None,
                                'negative_bars_mean': float,
                                'positive_bars_mean': float,
                                'max_dd': float,
                                'volatility': float,
                                'good_for_trading': bool
                                }

        """

        parts_length = len(entire_data) // self.number_of_chunks

        usability_result = {}
        for i in range(self.number_of_chunks):
            data_chunk = entire_data.iloc[i * parts_length:parts_length + (i * parts_length)]

            # Calculations related to bar movements
            chunk_results = \
                self.calculate_max_single_bar_gain_loss(data_chunk, calculate_bar_movements_using_open_to_close)

            # Max Draw Down Calculations

            chunk_results['max_dd'] = self.calc_max_dd(data_chunk[self.default_column_name])

            chunk_results['volatility'] = self.calculate_volatility(data_chunk[self.default_column_name])

            chunk_results['good_for_trading'] = chunk_results['volatility'] > self.min_data_volatility

            # i+1 is used to represent first part at 1 instead of 0
            usability_result['part_' + str(i + 1)] = chunk_results

            # Calculation for the entire data should be done again (so we have to call the following function on the
        # entire data:

        all_data_movement_results = self.calculate_max_single_bar_gain_loss(entire_data,
                                                                            calculate_bar_movements_using_open_to_close)

        all_data_movement_results['max_dd'] = self.calc_max_dd(entire_data[self.default_column_name])

        all_data_movement_results['volatility'] = self.calculate_volatility(entire_data[self.default_column_name])

        all_data_movement_results['good_for_trading'] = all_data_movement_results[
                                                            'volatility'] > self.min_data_volatility

        usability_result['total'] = all_data_movement_results

        return usability_result

    def generate_chunk_statistics(self, data_chunk,
                                calculate_bar_movements_using_open_to_close=True):  # , verbose=False):

        """
        This function assumes that the index for data is a date object
        :param calculate_bar_movements_using_open_to_close: This parameter determines if price movement is calculated
        using close-open or if it is calculated using the following formula:
        (previous self.default_column_name-current self.default_column_name)/previous self.default_column_name
        :param data_chunk: the chunk of dataframe containing all the information


        :return:  usability_dictionary = {'min_movement': float,
                                'max_movement': float,
                                'min_movement_date': None,
                                'max_movement_date': None,
                                'negative_bars_mean': float,
                                'positive_bars_mean': float,
                                'max_dd': float,
                                'volatility': float,
                                'good_for_trading': bool
                                }

        """

        # Calculations related to bar movements
        chunk_results = \
            self.calculate_max_single_bar_gain_loss(data_chunk, calculate_bar_movements_using_open_to_close)

        # Max Draw Down Calculations

        chunk_results['max_dd'] = self.calc_max_dd(data_chunk[self.default_column_name])

        chunk_results['volatility'] = self.calculate_volatility(data_chunk[self.default_column_name])

        chunk_results['good_for_trading'] = chunk_results['volatility'] > self.min_data_volatility


        return chunk_results

    def flatten(self, dictionary_, parent_key='', sep='_'):
        """
        This function flattens a nested dictionary using a recursive algorithm
        :param parent_key:
        :param dictionary_: original nested dictionary

        :param sep:
        :return:
        """
        items = []
        for k, v in dictionary_.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def convert_usability_result_to_df(self, usability_result):
        return pd.DataFrame(self.flatten(usability_result), index=[0])

# test_data = pd.DataFrame([[1, -23, 4, '2020-01-01', 0], [11, 223, 43, '2021-01-17', 63], [11, 13, 14, '2000-04-01', 0],
#                           [161, 2253, 443, '2018-01-07', 613], [38, 233, 430, '2020-10-01', 6],
#                           [38, 233, 4, '2000-10-01', 0]], columns=['A', 'open', 'close', 'date', 'E'])

# test_data.set_index('date', inplace=True)
# path = 'test_data'
# file_name = 'GRG1L.VS.csv'
# file_loader = LoadYahooSymbol()
# my_data = file_loader.load_file(path, file_name)
# # my_data = pd.read_csv('test_data/GRG1L.VS.csv', parse_dates=True)
#
# my_data.columns = my_data.columns.str.lower()
# print(my_data.iloc[10:20])
# my_util = Util('open', 'high', 'low', 'close', 'volume', 0.1, 3, 2, 1, 'close', min_number_of_sentences=2)
# my_data.index = pd.to_datetime(my_data.index, utc=True)
# print(pd.DataFrame(my_util.check_integrity_of_data(my_data.iloc[:10], path, file_name), index=[0]).T)
# # print(my_util.check_usability_of_data(my_data.iloc[:10]))
# results = my_util.check_usability_of_data(my_data.iloc[:10])
# print(my_util.convert_usability_result_to_df(results).iloc[0])
