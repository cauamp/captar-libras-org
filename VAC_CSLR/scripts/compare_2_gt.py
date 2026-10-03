import pandas as pd


zed_gt = "../datasets_srv/testes_cam_zed/gt.csv"
ids = "../gt_id_10-04.csv"

zgt = pd.read_csv(zed_gt)
zgt['sentence_id'] = zgt['sentence_id'].astype(str).replace('s', '', regex=True).str.strip()
print(f"{zgt.columns =}") # Index(['video', 'sentence_id', ' gloss'], dtype='object')
print(zgt.head())



gt = pd.read_csv(ids)
gt['idx'] = gt['id'].astype(str)
print(f"{gt.columns =}") # Index(['video', 'sentence_id', ' gloss'], dtype='object')
print(gt.head())

for i in range(len(zgt)):
    idx = zgt['sentence_id'][i]
    gloss_z = zgt['gloss'][i]
    value = gt.loc[gt['idx'] == idx]['gloss']   
    
    if gloss_z != value.values[0]:
        print(f"{gloss_z = }")
        print(f"{value.values[0] = }")
        print(f"{zgt['video'][i]}")
        print("-----------------------------------")

