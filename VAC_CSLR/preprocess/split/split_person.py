from pathlib import Path
import pandas as pd
import os

dev_ids = {49, 25, 40}
test_ids = {19, 33, 41}

def main(data: Path, excel: Path, consistent: Path, output: Path) -> None:
    os.makedirs(output, exist_ok=True)

    videos = {path.stem[:19]: path for path in sorted(data.rglob("*.avi"))}
    print(f"Number of recorded videos {len(videos)}")
    
    # region: Filter the spreadsheet.

    df = pd.read_excel(
        excel,
        sheet_name="Anotações",
        dtype={
            "video_id": str,
            "sentence_id": str,
            "person_id": int,
        },
        usecols=[
            "video_id",
            "person_id",
            "sentence_id",
            "question",
            "answer",
        ],
    ) 
    
    print(f'Number of videos in {excel}: {len(df)}')

    df["sentence_id"] = df["sentence_id"].str.strip()
    df["video_id"] = df["video_id"].str.strip()

    df["prefix"] = df["video_id"].str[:19]
    df["video_path"] = df["prefix"].map(videos)

    df["date"] = pd.to_datetime(df["prefix"], format="%d-%m-%Y_%H-%M-%S")

    df.sort_values("date")
    
    
    #print('Dates before filter:')
    #print(f'{df["date"].tail()}')	
    #df = df[df["date"] <= "2024-12-10"]
    #print('Dates after filter:')
    #print(f'{df["date"].tail()}')	
    
    exists = df["prefix"].isin(videos.keys())
    is_test = df["person_id"].isin({1, 2, 3, 4, 5, 6, 7, 8, 999})

    # region: Find missing videos.

    missing = df[~exists & ~is_test]
    missing = missing[["date", "sentence_id", "person_id", "video_id", "video_path"]]

    missing.to_csv(output / "missing.csv", index=False)
    print(f"Number of 'missing' videos (tests and non existing .avi): {len(missing)}")

    # endregion: Find missing videos.

    df = df[exists & ~is_test]
    df = df.dropna()

    # region: Find inconsistent videos.

    consistent = pd.read_csv(consistent, usecols=["VideoID"])
    consistent = consistent["VideoID"].str[:19]

    print("Number of videos before 'consistent' filter :", len(df))
    filtered = df[~df["prefix"].isin(consistent)]
    filtered.to_csv(output / "filtered.csv", index=False)
    
    df = df[df["prefix"].isin(consistent)]
    print("Number of videos after 'consistent' filter:", len(df))
    # endregion: Find inconsistent videos.

    df = df[["date", "sentence_id", "person_id", "video_id", "video_path"]]

    # endregion: Filter the spreadsheet.

    # region: Split data.

    n_classes = df["sentence_id"].nunique()

    # Print frequencies per sentence_id, sorted by frequency.
    #print('Frequencies per sentence_id, sorted by frequency:')
    #print(df.groupby("sentence_id").size().sort_values(ascending=False))


    train =    df[~df["person_id"].isin(dev_ids | test_ids)]
    test =     df[df["person_id"].isin(test_ids)]
    validate = df[df["person_id"].isin(dev_ids)]
    
    train.to_csv(output / "train.csv", index=False)
    validate.to_csv(output / "validate.csv", index=False)
    test.to_csv(output / "test.csv", index=False)

    #print(train.groupby("person_id").size())
    #print(validate.groupby("person_id").size())
    #print(test.groupby("person_id").size())
    
    print(f"Dataset size before 'signal' videos: {len(df)}")
    print(f"Train Size: {len(train)}")
    print(f"Test Size: {len(test)}")
    print(f"Validate Size: {len(validate)}")

    # endregion: Split data.


if __name__ == "__main__":
    data = '/srv/projects2/captarlibras_finep/dataset_captarlibras/raw-data/frontal-cam'
    exec = './[Captar-Libras] Lista de sentenças e pessoas informantes.xlsx'
    output = './2025-04-10'
    consistent = "../../datasets_/2025-04-10/rough_notes.csv"
    main(data=Path(data), excel=Path(exec), consistent=Path(consistent), output=Path(output))
