"""Constants describing the official FIGNEWS release schemas."""

FIGNEWS_CONTEXT_COLUMNS = ("batch", "source_language", "id", "type")
POST_KEY_COLUMNS = FIGNEWS_CONTEXT_COLUMNS
POST_UID_COLUMNS = ("source_language", "id")
POST_KEY_SEPARATOR = "::"
EXPECTED_PARTITIONS = frozenset({"MAIN", "IAA"})
EXPECTED_SOURCE_LANGUAGES = frozenset(
    {"Arabic", "English", "French", "Hebrew", "Hindi"}
)

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
    *FIGNEWS_CONTEXT_COLUMNS,
    "team_name",
    "sheet_type",
    "annotator_id",
    "task",
    "label",
    "notes",
)

SOURCE_RELEASE_COLUMNS = (
    "batch",
    "source_language",
    "id",
    "type",
    "text",
    "english_mt",
    "arabic_mt",
    "annotator_id",
    "bias",
    "propaganda",
    "comments",
)

SOURCE_REQUIRED_COLUMNS = SOURCE_RELEASE_COLUMNS[:7]
SOURCE_MODEL_COLUMNS = SOURCE_REQUIRED_COLUMNS

FLAT_HEADER_COLUMNS = (
    "batch",
    "source_language",
    "id",
    "type",
    "",
    "biased against palestine",
    "biased against israel",
    "biased against both palestine and israel",
    "biased against others",
    "unbiased",
    "not applicable",
    "unclear",
    "",
    "propaganda",
    "not propaganda",
    "not applicable",
    "unclear",
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
