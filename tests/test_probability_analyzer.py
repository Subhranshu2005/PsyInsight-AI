from psyinsight.probability.analyzer import ProbabilityAnalyzer


def main():

    values = [
        0.01,
        0.10,
        0.35,
        0.65,
        0.92,
    ]

    print("=" * 50)
    print("Probability Interpretation")
    print("=" * 50)

    for value in values:
        print(
            f"{value:.2f} --> "
            f"{ProbabilityAnalyzer.interpret(value)}"
        )


if __name__ == "__main__":
    main()
