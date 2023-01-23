import pandas as pd
import numpy as np
from line_printer import LinePrinter

ACTIONS = {-2: 'Explosive Short move',
           -1: 'All negative for the next number_of_bars_into_the_future',
           0: 'Mix Results',
           1: 'All positive for the next number_of_bars_into_the_future',
           2: 'Explosive Long move'}
# We could add other possibility such as closing positive or negative after number_of_bars_into_the_future
# this could help with Option selling
Trades = {-2: 'Short/Buy Put',
          -1: 'Sell Call',
          0: 'Flat',
          1: 'Sell Put',
          2: 'Long/Buy Call'}


class DataAggregator:

    def __init__(self, sentence_length, length_to_look_into_future_for_rewards,
                 explosive_move_look_back_period, min_volatility_for_explosive_move_filtering,
                 price_column_name, date_column_name, rounding_precision):
        """
        This class provides tools to aggregate historic and future data into a single row.
        It also sets action and rewards for the entire dataset. This system caps reward for explosive moves to the volatility
        level of the last explosive_move_look_back_period bars.
        In actual practice, it might be worth studying scenarios where optimal exit points can be determined rather than just
        capping the profit based on volatility of the last few bars

        :param sentence_length: the length of historic data to aggregate into one row
        :param length_to_look_into_future_for_rewards: number of bars to look into the future for calculating rewards and actions.
        this calculation start at the next bar and go all the way up to and including length_to_look_into_future_for_rewards
        :param explosive_move_look_back_period: number of bars to look back in order to calculate volatility of the stock.
        volatility is defined as: (max_-min_)/min_ over the explosive_move_look_back_period.
        If the price moves beyond this value in the next number_of_bars_into_the_future, then action will be set accordingly
        This parameter can also be called volatility_look_back_period
        :param min_volatility_for_explosive_move_filtering: this is the minimum volatility that the system will consider before categorizing the next movement as explosive
        This is used to fileter situation where the symbol moves in a very narrow band and then makes a move that is still very small but compared to the narrow band it might look large
        :param date_column_name: column name for date
        :param price_column_name: which column will be used for all the calculations (usually it is close)
        :param rounding_precision: precision level to store the data

        """

        self.sentence_length = sentence_length
        self.length_to_look_into_future_for_rewards = length_to_look_into_future_for_rewards
        self.explosive_move_look_back_period = explosive_move_look_back_period
        self.min_volatility_for_explosive_move_filtering = min_volatility_for_explosive_move_filtering
        self.rounding_precision = rounding_precision
        self.price_column_name = price_column_name
        self.date_column_name = date_column_name
        self.line_printer = LinePrinter()

    def gather_history(self, the_original_data_source):
        """
        :param the_original_data_source: dataframe to collect last self.sentence_length bars. The data must have date as index
        :return: a new dataframe where each row is self.sentence_length and the index is self.date_column_name
        """
        # todo: assert that _data has date as index

        # Transpose data
        sequence_data = the_original_data_source.T.loc[self.price_column_name].tolist()
        # place the data into chunks by iterating over the data
        the_end_ = len(sequence_data) - self.sentence_length + 1
        chunks = [sequence_data[x:x + self.sentence_length] for x in range(0, the_end_)]

        # Make a dataframe from the chunks
        chunk_df = pd.DataFrame(chunks)
        # Reverse the order of the columns name to place them from oldest to newest
        chunk_df.columns = chunk_df.columns[::-1]
        # Add a negative sign to the column names to indicate how many bars back we got the data from
        # todo: this method adds a - sign to columns 0 (the current row's column)
        ## its best to modify this in order to not add the - sign to column 0
        chunk_df = chunk_df.add_prefix("-")

        # in order to get the dates back, we merge the chunks with the original data Since chunk data only starts
        # after self.sentence_length bars, then we only merge _data starting at that position and -1 is to include
        # the current row in rese_index we keep drop=False since in our data, index is the date and we don't want to
        # lose that information
        merged = pd.merge(the_original_data_source.iloc[self.sentence_length - 1:][self.price_column_name].
                          reset_index(drop=False), chunk_df, right_index=True, left_index=True)

        # Drop the column name since the historic data at column 0 is the same as the current row's coumn_name data
        return merged.drop(self.price_column_name, axis=1).set_index(self.date_column_name)

    def calculate_returns(self, the_original_data_source):
        """
        This function calculates the return for the number of bars into the future
        It also calculates the sum of each row and weather all values are positive or negative (to determine the action to be taken)
        by calling the _calculate_total_return function
        :param the_original_data_source: the original dataframe

        """
        future_returns = pd.DataFrame()
        # Calculates return for the next number_of_bars_into_the_future
        for row in range(1, self.length_to_look_into_future_for_rewards + 1):
            col_ = str(row) + '_bar_into_future'
            future_returns[col_] = round((the_original_data_source[self.price_column_name].shift(-row) -
                                          the_original_data_source[self.price_column_name]) /
                                         the_original_data_source[self.price_column_name],
                                         self.rounding_precision)

        return future_returns

    def calculate_volatility(self, original_data_source):
        """

        :param original_data_source: the original datasource
        :return: (max_ - min_) / min_
        """
        min_ = original_data_source[self.price_column_name].rolling(self.explosive_move_look_back_period).min()
        max_ = original_data_source[self.price_column_name].rolling(self.explosive_move_look_back_period).max()
        return (max_ - min_) / min_

    def generate_volatility_matrix_by_comparing_next_bars_with_volatility(self, future_prices, volatility_):
        # todo: find a way to optimize this
        # We had to separate positive and negative volatility comparison dataframes because the formula for negative
        # movements was not the same as the one for the positive movement.
        # For the negative moves, we only consider trades where action is already set at -1
        # (price will not move above our entry point) where as for the long side, we don't check for this.
        # todo: it is best to make sure that the short side has the same entry condition as the long side

        volatility_df = future_prices[future_prices.columns[:self.length_to_look_into_future_for_rewards]].apply(
            lambda x: np.where(x > volatility_, 1, 0), axis=0).add_suffix('_exceeds_volatility')

        # volatility_df_for_negative = future_prices[future_prices['action'] == -1][
        #     future_prices.columns[:self.length_to_look_into_future_for_rewards]].apply(
        #     lambda x: np.where((1 / (1 + x)) > volatility_.loc[x.index.values], -1, 0), axis=0).add_suffix(
        #     '_exceeds_volatility')
        volatility_df_for_negative = future_prices[future_prices['action'] == -1][
            future_prices.columns[:self.length_to_look_into_future_for_rewards]].apply(
            lambda x: np.where(abs(x) > volatility_.loc[x.index.values], -1, 0), axis=0).add_suffix(
            '_exceeds_volatility')

        volatility_df.loc[volatility_df_for_negative.index.values] = volatility_df_for_negative

        volatility_df['original_volatility'] = volatility_

        return volatility_df

    def set_primary_actions_rewards(self, future_prices):
        """
        This function sets the primary actions and rewards.
        action 1 is set if all prices in the future prices dataframe are positive meaning the price did not go below
        our starting price.
        action -1 is set if all prices are negative meaning the price did not go above our entry point
        action 0 is set if the price crosses our entry price
        :param future_prices: is a Dataframe of all the returns calculated by calculate_returns function.
        :return: This function modifies the original future_prices dataframe and returns an update dataframe including
         action and rewards
        """

        all_negative = np.sign(future_prices) <= 0
        all_positive = np.sign(future_prices) >= 0
        # Sets the reward to the last bar we are checking in the future
        future_prices['reward'] = future_prices[future_prices.columns[-1]]
        future_prices['action'] = all_positive.all(axis=1).astype(int) - all_negative.all(axis=1).astype(int)

        # if the action is -1, reward should still be set as positive so this part changes the sign of reward
        future_prices.loc[future_prices['action'] == -1, 'reward'] = \
            future_prices.loc[future_prices['action'] == -1, 'reward'] * -1

        return future_prices

    def set_final_action_reward_including_explosive_moves(self, future_prices, volatility_matrix):
        """
        This function checks to see if price moved more than the volatility of the explosive_move_look_back_period bars
        it then sets explosive short moves, actions to -2 and long moves to 2
        it also caps the reward to the volatility level since that will be the target

        :param future_prices: first output of the calculate_returns function
        :param volatility_matrix: this is generated from calling generate_volatility_matrix_by_comparing_next_bars_with_volatility function

        :return: it returns the modified future_prices dataframe
        """

        future_prices['volatility'] = volatility_matrix['original_volatility']

        all_positive_ = np.sign(volatility_matrix[volatility_matrix.columns[:-1]].sum(axis=1)) > 0
        all_negative_ = np.sign(volatility_matrix[volatility_matrix.columns[:-1]].sum(axis=1)) < 0

        # todo: combine the following two assignment into one assignment
        future_prices.loc[(future_prices['volatility'] > self.min_volatility_for_explosive_move_filtering) &
                          all_positive_,
                          'action'] = 2

        future_prices.loc[(future_prices['volatility'] > self.min_volatility_for_explosive_move_filtering) &
                          all_negative_, 'action'] = -2

        return future_prices

    def aggregate_symbol_data(self, data_frame):
        history = self.gather_history(data_frame)
        future_returns = self.calculate_returns(data_frame)

        volatility = self.calculate_volatility(data_frame)

        primary_actions_and_rewards = self.set_primary_actions_rewards(future_returns)
        volatility_matrix = self.generate_volatility_matrix_by_comparing_next_bars_with_volatility(future_returns,
                                                                                                   volatility)
        final_action_and_rewards = self.set_final_action_reward_including_explosive_moves(future_returns,
                                                                                          volatility_matrix)
        final_action_and_rewards = pd.merge(history, final_action_and_rewards, left_index=True,
                                            right_index=True).dropna()

        return self.calculate_targets(final_action_and_rewards, volatility), volatility_matrix

    def calculate_targets(self, final_data_set, volatility_original):
        vol_values = volatility_original.loc[final_data_set.index.values]

        target_1 = vol_values.loc[final_data_set[final_data_set['action'] == 2].index.values]
        target_2 = round(vol_values.loc[final_data_set[final_data_set['action'] == -2].index.values],
                         self.rounding_precision)

        target_combined = pd.concat((target_1, target_2))
        target_combined.columns = ['target']

        final_data_set.loc[target_combined.index.values, 'target_percent'] = target_combined
        final_data_set.loc[final_data_set['action'] == -2, 'reward'] = final_data_set.loc[final_data_set['action']==-2, 'target_percent']
        final_data_set.loc[final_data_set['action'] == 2, 'reward'] = final_data_set.loc[
            final_data_set['action'] == 2, 'target_percent']


        return final_data_set


# test_data = pd.DataFrame([[1, 23, 4, '2020-01-01', 0], [11, 223, 43, '2021-01-17', 63], [11, 13, 14, '2000-04-01', 0],
#                           [161, 2253, 443, '2018-01-07', 613], [38, 233, 430, '2020-10-01', 6],
#                           [38, 233, 4, '2000-10-01', 0]], columns=['A', 'open', 'close', 'date', 'E'])
#
# test_data.set_index('date', inplace=True)

test_data = pd.read_csv('test_data/GRG1L.VS.csv', index_col=0)
# print(test_data)
data_aggregator = DataAggregator(16, 8, 20, 0.1, 'Close', 'Date', 4)
final_ds, vol_matrix = data_aggregator.aggregate_symbol_data(test_data)
print(final_ds[final_ds['target_percent'] > 0])
print(final_ds[final_ds['action'] == 1])
final_ds.to_csv('result.csv')
# vol_matrix.to_csv('vol_Matrix.csv')
# final_final = data_aggregator.calculate_targets(final_ds, data_aggregator.calculate_volatility(test_data))

print(final_ds[final_ds['action'] > 0])

# normalize data
# chart data and save the files
