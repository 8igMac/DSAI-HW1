import argparse
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def get_latest_data_as_df():
    '''Get latest data from Taiwan Power website.
    And return a pandas DataFrame.
    '''
    import requests
    res = requests.get('https://www.taipower.com.tw/d006/loadGraph/loadGraph/data/reserve.csv')

    from io import StringIO
    data_csv = StringIO(res.text)
    df = pd.read_csv(data_csv)
    return df

def split_train_test(data: pd.DataFrame, start: str, end: str, include=True):
    '''Extract features from data and split the data into training and 
    testing set.

    return: x_train, y_train, x_test, y_test
    '''
    # Some parameters.
    day_ahead = 2

    interested = data['備轉容量(MW)']

    shifts = np.arange(1, day_ahead + 1).astype(int)
    shifted_data = {f'lag_{day_shift}_day': interested.shift(day_shift) for day_shift in shifts}

    interested_shifted = pd.DataFrame(shifted_data)

    # Replace NaN with median.
    interested_shifted = interested_shifted.fillna(np.nanmedian(interested_shifted))
    interested = interested.fillna(np.nanmedian(interested))

    one_day = timedelta(days=1)
    fmt = '%Y%m%d'
    train_end = (datetime.strptime(start, fmt) - one_day).strftime(fmt)
    if include:
        test_end = end
    else:
        test_end = (datetime.strptime(end, fmt) - one_day).strftime(fmt)

    print(f'start: {start}, end: {end}, train_end: {train_end}, test_end: {test_end}')

    x_train = interested_shifted.loc[:train_end]
    y_train = interested.loc[:train_end]
    x_test = interested_shifted.loc[start:test_end]
    y_test = interested.loc[start:test_end]
        
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # Setup arguments.
    # 
    # You can use the arguments like this:
    # print(args.traning)
    # print(args.output)
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')

    parser.add_argument('--mode',
                        default='deploy',
                        help='test or deploy')
    args = parser.parse_args()

    if args.mode == 'deploy':
        start = '20220330'
        end = '20220413'
    else:
        start = '20220101'
        end = '20220116'

    # Load training data.
    data = pd.read_csv(args.training, index_col=0)

    # # TODO: Get current data.
    # df = get_latest_data_as_df()

    # TODO: Combine new data and training data.

    # Split the data into training and testing.
    x_train, y_train, x_test, y_test = split_train_test(data, start=start, end=end)

    # Train the model.
    model = Ridge()
    model.fit(x_train, y_train)

    # Predict the result. 
    pred = model.predict(x_test)

    # Evaluation.
    loss = mean_squared_error(y_test, pred, squared=False)
    print(f'Loss: {loss}')

    # Output the result to csv file.
    pred_df = pd.DataFrame({
        # Convert date format form 2022-03-31 to 20220331.
        'date': pd.date_range(start=start, end=end).strftime('%Y%m%d'),
        # Rounding and convert to int.
        'operating_reserve(MW)': pred.round(decimals=0).astype(int),
    })
    pred_df.to_csv(args.output, index=False)
