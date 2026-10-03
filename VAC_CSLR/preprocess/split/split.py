from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.contrib.logging import logging_redirect_tqdm


@click.command()
@click.option(
    "--data",
    type=Path,
    required=True,
    help="Caminho para a pasta que contém os vídeos de entrada."
)   
@click.option(
    "--excel",
    type=Path,
    required=True,
    help="Caminho para a planilha de anotações ([Captar-Libras] Lista de sentenças e pessoas informantes)."
)   
@click.option(
    "--test-size",
    type=int,
    default=2,
    help="Tamanho do conjunto de teste (definir como um número inteiro). Valor padrão é 2."
)   
@click.option(
    "--check-file",
    type=Path,
    default=None,
    help="Caminho para o arquivo .csv com vídeos verificados (ELAN x Planilha)."
)   
@click.argument(
    "output",
    type=Path,
)   
@logging_redirect_tqdm()
def main(
    data: Path, excel: Path, test_size: float, output: Path, check_file: Path
) -> None:
    videos = {path.stem[:19]: path for path in sorted(data.rglob("*.mp4"))}

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

    df = df[df["date"] >= "2024-08-22"]

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

    if check_file:
        checked = pd.read_csv(check_file, usecols=["VideoID"])
        checked = checked["VideoID"].str[:19]

        df = df[df["prefix"].isin(checked)]

    # endregion: Find inconsistent videos.

    df = df[["date", "sentence_id", "person_id", "video_id", "video_path"]]

    # endregion: Filter the spreadsheet.

    # region: Split data.

    n_classes = df["sentence_id"].nunique()

    # Print frequencies per sentence_id, sorted by frequency.
    print(df.groupby("sentence_id").size().sort_values(ascending=False))

    # Drop the sentences with less than `test_size` samples.
    df = df.groupby("sentence_id").filter(lambda x: len(x) >= 2 * test_size)

    if test_size == 0:
        train = df
        validate = pd.DataFrame(columns=df.columns)
        test = pd.DataFrame(columns=df.columns)
    else:
        train, test = train_test_split(
            df,
            test_size=(test_size * n_classes),
            stratify=df["sentence_id"],
            random_state=0,
        )
        train, validate = train_test_split(
            train,
            test_size=(test_size * n_classes),
            stratify=train["sentence_id"],
            random_state=0,
        )

    train = train.sort_values("date")
    test = test.sort_values("date")
    validate = validate.sort_values("date")

    train.to_csv(output / "train.csv", index=False)
    validate.to_csv(output / "validate.csv", index=False)
    test.to_csv(output / "test.csv", index=False)

    print(train.groupby("person_id").size())
    print(validate.groupby("person_id").size())
    print(test.groupby("person_id").size())

    # endregion: Split data.


if __name__ == "__main__":
    main()
