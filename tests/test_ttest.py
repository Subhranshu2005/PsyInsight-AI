from psyinsight.preprocessing.loader import DataLoader
from psyinsight.preprocessing.cleaner import DataCleaner
from psyinsight.statistics.inferential import InferentialStatistics


def main():

    df = DataLoader.load(
        "datasets/sample_psychology_data.csv"
    )

    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    cleaner.fill_missing_values()
    cleaner.rename_columns()

    clean_df = cleaner.get_dataframe()

    stats = InferentialStatistics(clean_df)

    result = stats.independent_t_test(
        column="stress_score",
        group_column="gender",
        group1="M",
        group2="F",
    )

    print("=" * 50)
    print("Independent t-Test")
    print("=" * 50)

    print(result)


if __name__ == "__main__":
    main()
