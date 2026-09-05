from psyinsight.probability.manual import ManualProbability
from psyinsight.probability.distributions import ProbabilityEngine


def main():

    k = 4
    lam = 3.5

    manual = ManualProbability.poisson_pmf(k, lam)

    scipy = ProbabilityEngine.poisson_probability(k, lam)

    print("=" * 50)
    print("Poisson Distribution Verification")
    print("=" * 50)

    print(f"Manual PMF : {manual:.12f}")
    print(f"SciPy PMF  : {scipy:.12f}")

    difference = abs(manual - scipy)

    print(f"Difference : {difference:.12e}")

    if difference < 1e-10:
        print("\n✅ Manual implementation verified successfully!")
    else:
        print("\n❌ Verification failed.")


def test_poisson_pmf_matches_scipy():
    k, lam = 4, 3.5
    manual = ManualProbability.poisson_pmf(k, lam)
    scipy_value = ProbabilityEngine.poisson_probability(k, lam)
    assert abs(manual - scipy_value) < 1e-10


if __name__ == "__main__":
    main()
