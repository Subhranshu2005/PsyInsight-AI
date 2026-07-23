import pandas as pd
from psyinsight.preprocessing.validator import DataValidator


def main():
    df = pd.read_csv("datasets/sample_psychology_data.csv")

    validator = DataValidator(df)
    report = validator.validate()

    print("\n===== DATA VALIDATION REPORT =====\n")

    for key, value in report.items():
        print(f"{key}:")
        print(value)
        print()


if __name__ == "__main__":
    main()
