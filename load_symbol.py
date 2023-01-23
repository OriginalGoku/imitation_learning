import pandas as pd
class LoadYahooSymbol:
    def load_file(self, path, file_name):
        file_format = file_name.split('.')[-1]
        try:
            # print("Trying for ", file_format)
            if file_format == 'csv':
                print('loading ', file_name)
                data = pd.read_csv(path+"/"+file_name, index_col=0, parse_dates=True)
            elif file_format == 'xlsx':
                data = pd.read_excel(path + "/" + file_name, index_col=0, parse_dates=True)
            else:
                raise Exception ('File format must be either .csv or .xlsx')

            data.index = pd.to_datetime(data.index, utc=True)

            return data
        except:
            print('Could not load ', path, '/', file_name)

