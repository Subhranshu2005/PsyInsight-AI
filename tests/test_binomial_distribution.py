from psyinsight.probability.manual import ManualProbability
from psyinsight.probability.distributions import ProbabilityEngine


def main():

    n = 10
    k = 8
    p = 0.7

    manual = ManualProbability.binomial_pmf(n, k, p)

    scipy = ProbabilityEngine.binomial_probability(k, n, p)

    print("=" * 50)
    print("Binomial Distribution Verification")
    print("=" * 50)

    print(f"Manual PMF : {manual:.12f}")
    print(f"SciPy PMF  : {scipy:.12f}")

    difference = abs(manual - scipy)

    print(f"Difference : {difference:.12e}")

    if difference < 1e-10:
        print("\n✅ Manual implementation verified successfully!")
    else:
        print("\n❌ Verification failed.")


if __name__ == "__main__":
    main()
