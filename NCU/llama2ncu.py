import os
import pandas as pd

os.system("ncu --csv --import ./llama2.ncu-rep > output.csv")

def preprocess_metric_unit(x):
    if 'dram__bytes' in x['Metric Name']:
        if x['Metric Unit'] == 'Gbyte':
            x['Metric Value'] = float(x['Metric Value']) * 1000
        elif x['Metric Unit'] == 'Kbyte':
            x['Metric Value'] = float(x['Metric Value']) / 1000
        elif x['Metric Unit'] == 'byte':
            x['Metric Value'] = float(x['Metric Value']) / 1000000
        else:
            x['Metric Value'] = float(x['Metric Value'])
        x['Metric Unit'] = 'Mbyte'
        return x
    elif 'time' in x['Metric Name']:
        if x['Metric Unit'] == 'second':
            x['Metric Value'] = float(x['Metric Value']) * 1000000
        elif x['Metric Unit'] == 'msecond':
            x['Metric Value'] = float(x['Metric Value']) * 1000
        else:
            x['Metric Value'] = float(x['Metric Value'])
        x['Metric Unit'] = 'usecond'
        return x
    elif 'cycles' in x['Metric Name']:
        x['Metric Value'] = x['Metric Value'].replace(',', '')
        return x
data = pd.read_csv('output.csv')
data=data.apply(lambda x : preprocess_metric_unit(x), axis=1)
pv_table = pd.pivot_table(data, index=['ID', 'Kernel Name'], columns='Metric Name', values='Metric Value', aggfunc='sum')
# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
print(pv_table)
pv_table.to_csv('llama2.csv')