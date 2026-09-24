# Verification report

The delivery was tested outside the Git repository against a locally generated
copy of `modeling_table_with_splits.parquet`. The split generator produced
10,800 train posts, 2,700 validation posts, and 1,500 held-out IAA test posts.

## Checks completed

- All Python files compile successfully.
- Unit and artifact-generation tests pass.
- The real FIGNEWS table exposes `text`, `english_mt`, and `arabic_mt`.
- Bias and propaganda both complete end-to-end using original multilingual text.
- Models, configuration, metrics, predictions, selection results, and confusion
  matrices are written to the expected artifact directories.

## Preliminary smoke-test results

These values verify that the code runs; they are not yet the final reported
experiment because the full language-comparison matrix has not been executed.

| Task | Selected settings | Validation macro F1 | IAA test macro F1 |
| --- | --- | ---: | ---: |
| Bias | C=2.0, word 1-2 grams | 0.4334 | 0.3618 |
| Propaganda | C=1.0, word 1-2 grams | 0.6606 | 0.6788 |

The weaker bias result is plausible because bias has five substantive classes,
whereas propaganda is binary. The next analytical step is to inspect per-class
recall and confusion matrices before changing the model.
