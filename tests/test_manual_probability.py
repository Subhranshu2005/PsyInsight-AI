from psyinsight.probability.manual import ManualProbability

result = ManualProbability.normal_pdf(
    x=6,
    mean=5,
    std=2
)

print(f"Normal PDF = {result}")
