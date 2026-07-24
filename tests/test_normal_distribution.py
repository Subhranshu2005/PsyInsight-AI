from psyinsight.probability.manual import ManualProbability
from psyinsight.probability.distributions import ProbabilityEngine


def main():

    x = 6
    mean = 5
    std = 2

    manual = ManualProbability.normal_pdf(x, mean, std)

    scipy = ProbabilityEngine.normal_probability(x, mean, std)

    print("=" * 50)
    print("Normal Distribution Verification")
    print("=" * 50)

    print(f"Manual PDF : {manual:.12f}")
    print(f"SciPy PDF  : {scipy:.12f}")

    difference = abs(manual - scipy)

    print(f"Difference : {difference:.12e}")

    if difference < 1e-10:
        print("\n✅ Manual implementation verified successfully!")
    else:
        print("\n❌ Verification failed.")


if __name__ == "__main__":
    main()
