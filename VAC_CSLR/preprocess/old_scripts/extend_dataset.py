from pathlib import Path
import pandas as pd


def main(data: Path, excel: Path, test_size: float, output: Path) -> None:
    videos = {path.stem[:19]: path for path in sorted(data.rglob("*.avi"))}

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

    df["sentence_id"] = df["sentence_id"].str.strip()
    df["video_id"] = df["video_id"].str.strip()

    df["prefix"] = df["video_id"].str[:19]
    df["video_path"] = df["prefix"].map(videos)

    df["date"] = pd.to_datetime(df["prefix"], format="%d-%m-%Y_%H-%M-%S")

    df = df[df["date"] <= "2024-08-22"]

    exists = df["prefix"].isin(videos.keys())
    is_test = df["person_id"].isin({1, 2, 3, 4, 5, 6, 7, 8, 999})

    # region: Find missing videos.

    missing = df[~exists & ~is_test]
    missing = missing[["date", "sentence_id", "person_id", "video_id", "video_path"]]

    missing.to_csv(output / "missing.csv", index=False)

    # endregion: Find missing videos.

    df = df[exists & ~is_test]
    df = df.dropna()


    # region: Find inconsistent videos.

    checked = pd.read_csv("./2024-10-16-checked/checked_29-11.csv", usecols=["VideoID"])
    checked = checked["VideoID"].str[:19]

    df = df[df["prefix"].isin(checked)]
    
    # region: Find inconsistent videos.
    
    df = df[["date", "sentence_id", "person_id", "video_id", "video_path"]]

    # endregion: Filter the spreadsheet.

    train = pd.read_csv("2024-08-22-checked/train.csv")
    test = pd.read_csv("2024-08-22-checked/test.csv")
    validate = pd.read_csv("2024-08-22-checked/validate.csv")
    base = pd.concat([train, test, validate])
    
    ctrl = df[~df["person_id"].isin(base["person_id"])]
    
    ctrl = ctrl.sort_values("date")
    ctrl.to_csv(output / "ctrl.csv", index=False)
    print(ctrl.groupby("person_id").size())

    # endregion: Split data.


if __name__ == "__main__":
    data = '/srv/projects2/captarlibras_finep/dataset_captarlibras/raw-data/frontal-cam'
    exec = './[Captar-Libras] Lista de sentenças e pessoas informantes.xlsx'
    output = './2024-10-16-checked'
    main(data=Path(data), excel=Path(exec), test_size=0.2, output=Path(output))