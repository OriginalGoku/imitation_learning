
import pandas as pd


class DataNormalizer:
    def __init__(self, clip=True, percentage_for_normalization=0.75, clip_max=1.5, clip_min=-0.5,
                 bin_size=0.1):
        # , std_clip_limit=2.5, second_part_std_multiplier=1.5):

        """
        This class normalizes the data. Normalization is made usnig the following formula: 1. subtract data from min
        of the first percentage_for_normalization*len(data) bars. 2. divide data to the maximum of the first
        percentage_for_normalization*len(data) bars This approach will cap the first percentage_for_normalization %
        of the data capped between 0 and 1 the rest of the data (1-percentage_for_normalization)% will not be capped
        and can go above 1 and below 0 in order to limit extreme moves, there is a clip parameter. if true,
        it will clip the maximum movement beyond 1 and 0

        :param clip: to clip or not to clip, thats the question. This parameter clips the maximum movement above 1 and below 0 to keep data in a range.

        :param percentage_for_normalization: the percentage of data to be used for normalization. a value of 1 means that the min and max will be calculated on the entire data


        """
        self.clip = clip
        self.percentage_for_normalization = percentage_for_normalization
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.bin_size = bin_size

    # Round Function:
    def my_round(self, x, precision=2):
        return round(self.bin_size * round(x / self.bin_size), precision)
    # Root Mean Square Error Calculator

    def calculate_rmse(self, test_df, truth_df):
        return ((test_df - truth_df) ** 2).mean() ** .5

    def normalize_data(self, _data, verbose=False):
        """
            :param _data: one column of a dataframe. The function will make a copy of this data in order to avoid modification to the original data
            :param verbose: determines if the function has to print statistics

            """
        data = _data.copy()
        length = len(data)
        normalization_length = int(self.percentage_for_normalization * length)
        if verbose:
            print("normalization_length: ", normalization_length)
            print("Length: ", length)
            print("percentage_for_normalization: ", self.percentage_for_normalization)
        # if self.clip:
        # clip_limit_upper_first_part = data.iloc[:normalization_length].mean() + (
        #         std_clip_limit * data.iloc[:normalization_length].std())
        # clip_limit_lower_first_part = data.iloc[:normalization_length].mean() - (
        #         std_clip_limit * data.iloc[:normalization_length].std())
        # data.iloc[:normalization_length] = np.clip(data.iloc[:normalization_length],
        #                                            clip_limit_lower_first_part, clip_limit_upper_first_part)
        # clip_limit_upper_second_part = data.iloc[:normalization_length].mean() + (
        #         std_clip_limit * second_part_std_multiplier * data.iloc[:normalization_length].std())
        # clip_limit_lower_second_part = data.iloc[:normalization_length].mean() - (
        #         std_clip_limit * second_part_std_multiplier * data.iloc[:normalization_length].std())
        # data.iloc[normalization_length:] = np.clip(data.iloc[normalization_length:],
        #                                            clip_limit_lower_second_part, clip_limit_upper_second_part)



        min_data = (data.iloc[:normalization_length]).min()
        if verbose:
            print("min_data: ", min_data)
        if min_data > 0:
            minimized = data - min_data
        # this is in case our data has negative values.
        # This could be checked in another code to disregard data with negative values but this code is used as
        # a protection mechanism just in case.
        else:
            minimized = data + abs(min_data)

        if verbose:
            print("minimized: \n", minimized)

        max_data = minimized.iloc[:normalization_length].max()
        if verbose:
            print("max_data: ", max_data)
            print("normalization_length: ", normalization_length)

        normalized_data = minimized / max_data
        if self.clip:
            normalized_data = normalized_data.clip(self.clip_min, self.clip_max)

        return normalized_data

    def place_data_into_bins(self, normalized_data):
        return self.my_round(normalized_data)


# Old code:
# This was the original normalization function that included dynamic clipping. It also included clipping for the first part of the data

# def _normalize_data(_data, clip: bool, percentage_for_normalization=1, std_clip_limit=2.5,
#                    second_part_std_multiplier=1.5, verbose=False):
#     """
#     :param data: one column of a dataframe. The function will make a copy of this data in order to avoid modification to the original data
#     :param clip: to clipt or not
#     :param percentage_for_normalization: the percentage of data to be used for normalization. a value of 1 means that the min and max will be calculated on the entire data
#     :param std_clip_limit: the system will clip the first part of [:(len(data)*percentage_for_normalization)] and then will clip any value in this range, outside the boundaries of the mean+-std_clip_limit
#     :param second_part_std_multiplier: the second part of data [(len(data)*percentage_for_normalization):] will be clipped using the first part std_clip_limit*second_part_std_multiplier thus allowing further freedom for price to move in the second part of the data
#     """
#     data = _data.copy()
#     length = len(data)
#     normalization_length = int(percentage_for_normalization * length)
#     if verbose:
#         print("normalization_length: ", normalization_length)
#         print("Length: ", length)
#         print("percentage_for_normalization: ", percentage_for_normalization)
#     if clip:
#         clip_limit_upper_first_part = data.iloc[:normalization_length].mean() + (
#                 std_clip_limit * data.iloc[:normalization_length].std())
#         clip_limit_lower_first_part = data.iloc[:normalization_length].mean() - (
#                 std_clip_limit * data.iloc[:normalization_length].std())
#         data.iloc[:normalization_length] = np.clip(data.iloc[:normalization_length], clip_limit_lower_first_part,
#                                                    clip_limit_upper_first_part)
#         clip_limit_upper_second_part = data.iloc[:normalization_length].mean() + (
#                 std_clip_limit * second_part_std_multiplier * data.iloc[:normalization_length].std())
#         clip_limit_lower_second_part = data.iloc[:normalization_length].mean() - (
#                 std_clip_limit * second_part_std_multiplier * data.iloc[:normalization_length].std())
#         data.iloc[normalization_length:] = np.clip(data.iloc[normalization_length:], clip_limit_lower_second_part,
#                                                    clip_limit_upper_second_part)
#
#     min_data = data.iloc[:normalization_length].min()
#     if verbose:
#         print("min_data: ", min_data)
#     if min_data > 0:
#         minimized = data - min_data
#     else:
#         minimized = data + abs(min_data)
#     if verbose:
#         print("minimized: \n", minimized)
#     max_data = minimized.iloc[:normalization_length].max()
#     if verbose:
#         print("max_data: ", max_data)
#         print("normalization_length: ", normalization_length)
#
#     normalized_data = minimized / max_data
#     return normalized_data


#
#
# norm = DataNormalizer(bin_size=0.1)
# tes = pd.DataFrame([[1,2,3,4,5],[3,5,1,5,9],[10,19,33,34,15],[11,12,13,11,15]],columns='A B C D E'.split())
# print(tes.A)
# normalized_data_ = norm.normalize_data(tes.A)
# print(normalized_data_)
# print(norm.place_data_into_bins(normalized_data_))