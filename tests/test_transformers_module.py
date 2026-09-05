from psyinsight.transformers import TextAnalyzer

TEXTS = [
    "I feel really happy and grateful today",
    "This has been a terrible and stressful week",
    "It was an ordinary day, nothing special happened",
]


def test_sentiment_returns_expected_columns():
    analyzer = TextAnalyzer(use_transformers=False)
    result = analyzer.sentiment(TEXTS)
    assert set(result.columns) == {"text", "label", "score"}
    assert len(result) == len(TEXTS)


def test_sentiment_summary_distribution_sums_to_one():
    analyzer = TextAnalyzer(use_transformers=False)
    summary = analyzer.sentiment_summary(TEXTS)
    assert abs(sum(summary["distribution"].values()) - 1.0) < 1e-2


def test_extract_keywords_returns_dataframe():
    keywords = TextAnalyzer.extract_keywords(TEXTS, top_k=5)
    assert "keyword" in keywords.columns
    assert len(keywords) <= 5


def test_discover_themes_partitions_all_responses():
    analyzer = TextAnalyzer(use_transformers=False)
    result = analyzer.discover_themes(TEXTS, n_themes=2)
    total = sum(info["n_responses"] for info in result["themes"].values())
    assert total == len(TEXTS)
