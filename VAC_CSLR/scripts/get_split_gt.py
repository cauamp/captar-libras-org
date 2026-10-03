import pandas as pd
import csv

path = '../datasets_srv/captar-libras_10-04/annotations/train.csv'


df = pd.read_csv(path, sep='|')

gt = {}
print(df.columns)
for i in range(len(df)):
    if '_signal' in df['id'][i]:
        continue
    id = df['id'][i].split('_s')[-1].split('_')[0]
    
    if id not in gt:
        gt[id] = df['annotation'][i]
    else:
        if gt[id] != df['annotation'][i]:
            print(f'ID {id} has different annotations: {gt[id]} and {df["annotation"][i]}')

gt = dict(sorted(gt.items(), key=lambda x: float(x[0].split('P')[-1]) if 'P' in x[0] else float(x[0])))
with open('gt.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'annotation']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for id, annotation in gt.items():
        writer.writerow({'id': id, 'annotation': annotation})

