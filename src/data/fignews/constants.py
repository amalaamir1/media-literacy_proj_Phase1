"""Constants describing the official FIGNEWS release schemas."""

POST_KEY_COLUMNS = ("batch", "source_language", "id", "type")

BIAS_LABELS = (
    "biased against palestine",
    "biased against israel",
    "biased against both palestine and israel",
    "biased against others",
    "unbiased",
    "not applicable",
    "unclear",
)

PROPAGANDA_LABELS = (
    "propaganda",
    "not propaganda",
    "not applicable",
    "unclear",
)

SUBSTANTIVE_LABELS = {
    "bias": frozenset(BIAS_LABELS[:5]),
    "propaganda": frozenset(PROPAGANDA_LABELS[:2]),
}

UNCERTAIN_LABELS = frozenset({"unclear"})
NOT_APPLICABLE_LABELS = frozenset(
    {"not applicable", "not_applicable", "n/a", "na"}
)

LONG_REQUIRED_COLUMNS = (
    *POST_KEY_COLUMNS,
    "subtask",
    "label",
    "text",
)

SOURCE_REQUIRED_COLUMNS = (
    "batch",
    "source_language",
    "id",
    "type",
    "text",
)

FLAT_POSITIONAL_COLUMNS = (
    "batch",
    "source_language",
    "id",
    "type",
    "_bias_separator",
    "bias__biased_against_palestine",
    "bias__biased_against_israel",
    "bias__biased_against_both_palestine_and_israel",
    "bias__biased_against_others",
    "bias__unbiased",
    "bias__not_applicable",
    "bias__unclear",
    "_propaganda_separator",
    "propaganda__propaganda",
    "propaganda__not_propaganda",
    "propaganda__not_applicable",
    "propaganda__unclear",
)