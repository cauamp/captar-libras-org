from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.metrics import Metric
import numpy as np


class WER(Metric):
    def __init__(self, pre_name="", output_transform=lambda x: x):
        self.erros = None
        self.total = None
        self.wer_list = None
        if pre_name and not pre_name.endswith("_"):
            pre_name += "_"
        self.pre_name = pre_name
        super(WER, self).__init__(output_transform=output_transform)

    def wer(self, original, recognized):
        """Calculate Word Error Rate (WER) between original and recognized text.

        Uses dynamic programming to compute the minimum number of word-level operations
        (insertions, deletions, substitutions) needed to transform the original text
        into the recognized text.

        Args:
            original (str): The original/reference text
            recognized (str): The recognized/hypothesis text

        Returns:
            tuple: A tuple containing:
                - wer_value (float): WER as a percentage (number of operations / length of original)
                - e (int): Total number of operations needed
                - t (int): Length of original text in words
        """
        words_original = original.split()
        words_recognized = recognized.split()

        dp = [[0] * (len(words_recognized) + 1)
              for _ in range(len(words_original) + 1)]

        for i in range(len(words_original) + 1):
            for j in range(len(words_recognized) + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif words_original[i - 1] == words_recognized[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1],
                                       dp[i - 1][j], dp[i][j - 1])

        wer_value = (
            dp[len(words_original)][len(words_recognized)] *
            100 / len(words_original)
        )

        e = dp[len(words_original)][len(words_recognized)]
        t = len(words_original)
        return wer_value, e, t

    @reinit__is_reduced
    def reset(self):
        self.erros = 0
        self.total = 0
        self.wer_list = []

        super(WER, self).reset()

    @reinit__is_reduced
    def update(self, output):
        preds, targets = output
        # targets is a list ["a is an apple","b is a banana"]
        # pred is a list ["what is an apple", "b is a banana"]

        for pred, target in zip(preds, targets):
            wer_value, e, t = self.wer(target, pred)
            self.erros += e
            self.total += t
            self.wer_list.append(wer_value)

    @sync_all_reduce(
        "erros",
        "total",
        "wer_list"
    )
    def compute(self):
        scores = {
            f"{self.pre_name}wer_primary": self.erros / self.total * 100,
            f"{self.pre_name}wer_mean": np.mean(self.wer_list),
            f"{self.pre_name}wer_std": np.std(self.wer_list),
            f"{self.pre_name}e": self.erros,
            f"{self.pre_name}t": self.total,

            f"{self.pre_name}wer_list": self.wer_list,
        }
        return scores
