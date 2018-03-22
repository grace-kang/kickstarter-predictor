import numpy
import pandas
from sklearn.model_selection import train_test_split

def balancing_info(df):
    succ = 0
    fail = 0
    for row in df:
        if row[0] == 0:
            fail += 1
        else:
            succ += 1
    return succ, fail

dataset = pandas.read_csv('kickstarter_data_full.csv', low_memory = False)
print('== Size of Original Dataset ==')
print(str(dataset.shape) + '\n')

dataset = dataset[[
    'blurb',
    'launch_to_deadline_days',
    'backers_count',
    'goal',
    'disable_communication',
    'country',
    'currency',
    'staff_pick',
    'static_usd_rate',
    'category',
    'SuccessfulBool']]
print('== Size of Dataset After Feature Extraction ==')
print(str(dataset.shape) + '\n')

dataset = dataset.dropna(axis=0, how='any')
dataset = dataset[dataset.country != 'LU'] # There is only one from LU.
print('== Size of Dataset After Removing Nulls ==')
print(str(dataset.shape) + '\n')

dataset_1, dataset_2 = train_test_split(dataset, test_size=0.5, random_state = 42)
print('== Size of Dataset 1 ==')
print(str(dataset_1.shape) + '\n')
print('== Size of Dataset 2 ==')
print(str(dataset_2.shape) + '\n')

succ, fail = balancing_info(dataset_1[['SuccessfulBool']].values)
print('== Balancing Info of Dataset 1 ==')
print('Number Successful: ' + str(succ) + ', Number Fail: ' + str(fail))
print('Percent Successful: ' + str(succ / (succ + fail)))
print('===============================\n')

succ, fail = balancing_info(dataset_2[['SuccessfulBool']].values)
print('== Balancing Info of Dataset 2 ==')
print('Number Successful: ' + str(succ) + ', Number Fail: ' + str(fail))
print('Percent Successful: ' + str(succ / (succ + fail)))
print('===============================\n')

print('Writting Datasets to CSVs.')
dataset_1.to_csv('test_data.csv', encoding='utf-8', index=False)
dataset_2.to_csv('train_data.csv', encoding='utf-8', index=False)
