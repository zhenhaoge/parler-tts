# prepare the data manifest json file for the demo data
#
# the data json file contains the 1) prompt text, 2) description text, and 3) attributes (optional) 
#
# Zhenhao Ge, 2024-05-08

import os
import argparse
import pandas as pd
import json

def parse_args():
    usage = 'usage: prepare the data json file for the demo data'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--excel-file', type=str, help='manifest excel file')
    parser.add_argument('--json-file', type=str, help='manifest json file')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()
    # data_dir = '/home/users/zge/code/repo/parler-tts/examples/parler-tts-demo'
    # args.excel_file = os.path.join(data_dir, 'manifest.xlsx')
    # args.json_file = os.path.join(data_dir, 'manifest.json')

    assert os.path.isfile(args.excel_file), \
        'manifest excel file: {} does not exist!'.format(args.excel_file)

    # read the data frame from the input manifest excel file
    df = pd.read_excel(args.excel_file)
    num_entries = len(df)
    print('# of entries in {}: {}'.format(os.path.basename(args.excel_file), num_entries))

    # get the entries
    entries = [{} for _ in range(num_entries)]
    fields = list(df.keys())
    num_fields = len(fields)
    for i in range(num_entries):
        for j in range(num_fields):
            entries[i][fields[j]] = df[fields[j]][i]

    # save the output manifest json file
    with open(args.json_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    print('wrote manifest json file: {}'.format(args.json_file))           
