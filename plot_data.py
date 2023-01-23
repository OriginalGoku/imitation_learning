import line_printer
import pandas as pd
from load_symbol import LoadYahooSymbol
import os
from statistic_generator import StatisticGenerator
from line_printer import LinePrinter
from data_plotter import Plotter
import random
from tqdm import tqdm
import timeit

MAX_DATE_GAP = 10
MIN_DATA_VOLATILITY = 0.1
MIN_NUMBER_OF_SENTENCES_TO_CONSIDER_VALID_DATA = 10
ROUNDING_PRECISION = 2


class PlotData:
    def __init__(self, percentage_of_data_to_plot: float, use_different_path_to_save_data: bool, mean_length: int,
                 column_name_for_plotting, plot_title, min_length_of_data_to_plot,
                 sentence_length: int, future_length: int, use_sentence_length_for_data_splitting: bool,
                 data_splits_for_plotting=3,
                 save_path=None, path="Yahoo_Stock", file_formats_to_load='csv', save_plots=True, verbose=False,
                 clip=True, percentage_for_normalization=0.75, clip_max=1.5, clip_min=-0.5, bin_size=0.1):
        """
        This function plots a percentage of the entire data and saves them in their own directory.
        :param percentage_of_data_to_plot: % of total data to plot. maximum is 1
        :data_splits_for_plotting: this parameter determines how many parts data is split for printing. a good number
        should be 2 or 3
        :param path:
        :param file_formats_to_load:
        :param sentence_length: The actual size of input (number of bars) we want to give to our learning system
        :param future_length: The number of bars to look into future for generating results
        :param use_sentence_length_for_data_splitting: if True, it will use sentence length to plot the data.
        Also, if true, then the percentage_for_normalization will be calculated as:
        (sentence_length)/(sentence_length+future_length)
        """

        if use_sentence_length_for_data_splitting:
            percentage_for_normalization = sentence_length / (sentence_length + future_length)

        if "/" != path[-1]:
            raise Exception("Path name must end with /")

        if percentage_of_data_to_plot > 1:
            raise ValueError("percentage_of_data_to_plot must be less than 1")

        plot_save_path = path
        if use_different_path_to_save_data:
            if not save_path:
                raise Exception("Have to provide a save path if use_different_path_to_save_data is True")
            elif not os.path.isdir(save_path):
                os.makedirs(save_path)
            plot_save_path = save_path

        self.path = path
        self.file_formats_to_load = file_formats_to_load
        self.percentage_of_data_to_plot = percentage_of_data_to_plot
        self.folder_list = self.load_all_sub_directories()

        self.plotter = Plotter(mean_length, plot_title, save_plots=save_plots, verbose=verbose, clip=clip,
                               percentage_for_normalization=percentage_for_normalization, clip_max=clip_max,
                               clip_min=clip_min, bin_size=bin_size)

        self.file_loader = LoadYahooSymbol()
        self.data_splits_for_plotting = data_splits_for_plotting
        if use_different_path_to_save_data:
            self.save_path = plot_save_path
        else:
            self.save_path = self.path
        self.line_printer = LinePrinter("-")
        self.column_name_for_plotting = column_name_for_plotting
        self.min_length_of_data_to_plot = min_length_of_data_to_plot
        statistics_input = {'open_column_name': 'Open', 'high_column_name': 'High', 'low_column_name': 'Low',
                            'close_column_name': 'Close', 'volume_column_name': 'Volume',
                            'min_data_volatility': MIN_DATA_VOLATILITY,
                            'sentence_length': sentence_length, 'future_data_length': future_length,
                            'max_date_gap': MAX_DATE_GAP,
                            'default_column_name': 'Close',
                            'min_number_of_sentences': MIN_NUMBER_OF_SENTENCES_TO_CONSIDER_VALID_DATA,
                            'rounding_precision': ROUNDING_PRECISION}
        self.statistic_generator = StatisticGenerator(**statistics_input)
        self.sentence_length = sentence_length
        self.future_length = future_length
        self.use_sentence_length_for_data_splitting = use_sentence_length_for_data_splitting

    def load_all_sub_directories(self):
        all_folders = os.listdir(self.path)
        folder_list = []

        for file in all_folders:
            if os.path.isdir(self.path + file):
                folder_list.append(file)

        return folder_list

    def get_file_names_in_directory(self, dir_):
        all_files = os.listdir(self.path + dir_)
        file_list = []
        print(self.path + dir_)
        for file in all_files:
            if os.path.isfile(self.path + dir_ + "/" + file):
                if file.split('.')[-1] == self.file_formats_to_load:
                    file_list.append(file)
        return file_list

    def plotter_all_data(self):
        statistic_results = []
        for i in tqdm(range(len(self.folder_list))):
            folder = self.folder_list[i]
            files_in_directory = self.get_file_names_in_directory(folder)
            # self.line_printer.print_line(text = str(int(len(files_in_directory) * self.percentage_of_data_to_plot)))
            for file_counter in tqdm(range((int(len(files_in_directory) * self.percentage_of_data_to_plot)))):
                # file_counter = random.randint(0, len(files_in_directory) - 1)
                load_file_path = self.path + folder + "/"
                save_file_path = self.save_path + folder + "/"
                file_name = files_in_directory[file_counter]
                file_data = self.file_loader.load_file(load_file_path, file_name)

                singe_file_statistics = {}

                file_data_length = len(file_data)
                if file_data_length > self.min_length_of_data_to_plot:
                    singe_file_statistics['file_name'] = file_name
                    singe_file_statistics['symbol'] = file_name.replace('.csv', '')

                    if self.use_sentence_length_for_data_splitting:
                        date_step_size = self.sentence_length+self.future_length
                        number_of_chunks = file_data_length//date_step_size
                        print("Each part of plot has ", date_step_size, ' Bars')

                    else:
                        date_step_size = file_data_length // self.data_splits_for_plotting
                        number_of_chunks = self.data_splits_for_plotting
                        print("Each part of plot has ", date_step_size, ' Bars')


                    for chunk_id in range(number_of_chunks):
                        start_range = chunk_id * date_step_size
                        end_range = chunk_id * date_step_size + date_step_size
                        chunk_data = file_data.iloc[start_range:end_range]

                        # Statistics
                        print('Loading Statistics for ', file_name, ' part ', chunk_id)
                        singe_file_statistics[
                            'part_' + str(chunk_id + 1)] = self.statistic_generator.generate_chunk_statistics(
                            chunk_data)

                        usability_result = singe_file_statistics['part_' + str(chunk_id + 1)]['good_for_trading']

                        print("Plotting Part ", chunk_id, " of ", file_name)
                        usability_text = ''
                        if not usability_result:
                            usability_text = usability_text + 'NOT_'
                        usability_text = usability_text + 'Usable_'

                        plot_title = "Part_" + str(chunk_id + 1) + "_" + usability_text + str(date_step_size) + \
                                     "_bars_" + file_name
                        self.plotter.plot_values(chunk_data[self.column_name_for_plotting], save_file_path, plot_title,
                                                 x_step_size=len(chunk_data) // 10)

                    print('Loading Statistics for ', file_name, ' Total')
                    singe_file_statistics['total'] = self.statistic_generator.generate_chunk_statistics(file_data)

                    statistic_results.append(self.statistic_generator.flatten(singe_file_statistics))

        result_df = pd.DataFrame(statistic_results)
        result_df.set_index(result_df['symbol'], drop=True, inplace=True)
        result_df.drop('symbol', axis=1, inplace=True)
        result_df.to_csv(self.path + 'Statistics.csv')


my_plotter = PlotData(1, True, 20, 'Close', 'Comparing Price and Normalized Price', 100, 64, 16, True,
                      save_path='charts/', path="test_data/")
folders = my_plotter.load_all_sub_directories()
print(folders)
print(my_plotter.get_file_names_in_directory(folders[0]))
starting_time = timeit.default_timer()
print("Start time :", starting_time)
my_plotter.plotter_all_data()
print("Time difference :", timeit.default_timer() - starting_time)
