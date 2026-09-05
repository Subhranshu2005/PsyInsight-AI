from psyinsight.probability.manual import ManualProbability

result = ManualProbability.normal_pdf(
    x=6,
    mean=5,
    std=2
)

print(f"Normal PDF = {result}")


def test_normal_pdf_is_positive():
    value = ManualProbability.normal_pdf(x=6, mean=5, std=2)
    assert value > 0
