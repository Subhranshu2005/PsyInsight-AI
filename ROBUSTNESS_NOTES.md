# Robustness pass -- summary of changes

Two rounds of work. Round 1 fixed broken app wiring and added basic
hardening. Round 2 (this update) did a systematic API sweep of the large
ML modules, refactored the plotting layer for real thread-safety, added
property-based fuzz testing, and added application-level hardening
(access gate, rate limiting, upload content-sniffing).

## Round 1 -- real bugs fixed

1. **Classification auto-train was broken in the UI** -- `KeyError` on
   every use (`PsyClassifier.auto_train()` returned a different schema
   than the UI expected). Fixed by unifying the return schema.
2. **Independent t-test was called with the wrong arguments** -- `TypeError`
   on every use (2 args passed, 4 required). Fixed the UI wiring.
3. **Clustering's own fallback path re-raised the same exception it just
   caught.** Fixed by removing the fallback and adding proper pre-checks.
4. **Single-class classification crashed with a raw sklearn traceback.**
   Fixed with a clear pre-check.
5. **`rename_columns()` crashed on integer/mixed-type column headers.**
   Fixed by coercing headers to `str` first.
6. **`detect_straight_lining()` crashed on an empty dataframe.** Fixed to
   return a correctly-shaped empty result.

Plus: upload size/row/column limits, `DataValidator`/`DataCleaner` type
guards, a 0/0->NaN fix in missing-value percentage, and per-tab error
isolation in the Streamlit app.

## Round 2 -- systematic sweep of the large ML modules

Rather than a literal line-by-line read of `clustering.py` (3,450+ lines)
and `classifier.py` (1,200 lines), I wrote a script that introspects every
public method on a fitted engine and calls it with synthesized arguments,
logging every exception. That surfaced real bugs far faster than manual
reading would have, and each was verified in isolation before fixing:

7. **`PsyClustering.benchmark()` silently discarded the user's active
   model.** Documented as a read-only timing diagnostic ("time fit() for
   each algorithm"), it actually left whichever algorithm it timed *last*
   as the permanently active model -- so calling `benchmark()` after
   fitting your real model would silently swap it out. Fixed with a
   save/restore around the loop; `benchmark()` is now genuinely read-only.
8. **`PsyClustering.contingency()` was broken whenever the discovered
   cluster count differed from the true class count** (the normal case in
   clustering). It used `sklearn.metrics.confusion_matrix`, which builds a
   *square* matrix over the union of both label sets -- so a table meant to
   be `(n_true_classes, n_clusters)` came back as `(n_union, n_union)` and
   crashed trying to force it into the wrong-shaped DataFrame. Fixed by
   switching to `sklearn.metrics.cluster.contingency_matrix`, the function
   actually designed to produce rectangular contingency tables.
9. **`PsyClassifier.compare_models()` had no fault isolation between
   candidates** -- unlike `PsyRegressor`, one bad candidate (e.g. KNN
   needing more neighbors than available training samples) crashed the
   *entire* comparison, taking `auto_train()` down with it. Fixed to match
   the regressor's per-candidate try/except pattern: a failing candidate is
   now skipped (with a warning) instead of stopping everything else.
10. **Both `compare_models()` implementations raised a bare, confusing
    `KeyError: 'r2'` (or similar) if literally every candidate failed to
    fit** (e.g. an all-NaN target). Fixed to raise a clear `ValueError`
    explaining what happened and listing per-candidate failure reasons.

`PsyClassifier.compare_models()`/`PsyClustering.compare_clusters()`
themselves intentionally leave the *last-iterated* candidate as the active
model (by design, paired with `best_model()`/`best_clustering()` to load
the actual winner) -- this is documented behavior, not a bug, and is
different from the `benchmark()` issue above.

## Round 2 -- matplotlib thread-safety, actually fixed this time

Previously mitigated with a lock around the app's plotting call sites.
That protected the app's own UI but didn't make the `Visualizer` class
itself safe for concurrent/library use, since it still drew onto
matplotlib's global `pyplot` state (`plt.figure()`, `plt.gcf()`).

`src/psyinsight/visualization/visualizer.py` has been rewritten to use
matplotlib's object-oriented API: every method builds its own
`matplotlib.figure.Figure` via `Figure()` / `fig.add_subplot()` and
**returns** it, rather than mutating shared global state. Two independent
plot calls now produce two independent `Figure` objects, so concurrent
Streamlit sessions genuinely cannot interfere with each other -- verified
with a test that hammers the class from 16 threads x 100 calls
simultaneously (`tests/test_visualizer.py::TestVisualizerConcurrency`).
The app-level lock and `_show_fig()` helper were removed as no longer
necessary; call sites now do `st.pyplot(viz.histogram(...))` directly.

`Visualizer` also gained the same `TypeError`-on-bad-input guard as
`DataValidator`/`DataCleaner`, and every method now handles empty/all-NaN
data without crashing (previously untested).

## Round 2 -- property-based fuzz testing

`tests/test_fuzz.py` uses Hypothesis to generate hundreds of randomized
(including degenerate/malformed) dataframes and inputs per test run, and
asserts a broad property: **no unexpected exception type escapes**. A
clear `ValueError`/`TypeError`/`InvalidClusterError` is fine; an
`AttributeError`, `KeyError`, `IndexError`, or similar bubbling up from
inside a method is exactly the class of bug this whole pass has been
hunting, and randomized generation is a much higher-yield way to find more
of them than hand-picking cases. Covers `DataValidator`, `DataCleaner`,
`DescriptiveStatistics`, `ProbabilityEngine`, `Visualizer`, and the
classifier/regressor/clustering `auto_train`/`fit_predict` paths. At 150
examples per test (an ad hoc deeper run beyond the default), no new bugs
turned up beyond what the earlier targeted sweep had already found and
fixed -- some evidence, not proof, that the sweep's fixes covered the
higher-probability failure modes.

## Round 2 -- application-level hardening

- **Optional password gate.** If an operator sets `app_password` in
  Streamlit secrets, the app requires it before rendering anything.
  Unconfigured (the default), it's a complete no-op so existing/local
  usage is unaffected. This is a basic shared-password gate, not a full
  identity/auth system -- no per-user accounts, no session expiry beyond
  the browser tab, and it relies on the deployment's own TLS for transport
  security (outside this application's control; see "what remains
  infra-level" below).
- **Per-session rate limiting.** Training and clustering are now
  throttled to at most once per 3 seconds per browser session
  (`rate_limit_ok()`), so one user mashing a button can't monopolize CPU
  on a process shared by other users. Also an application-level
  mitigation, not a substitute for a reverse-proxy/WAF rate limiter.
- **Upload content-sniffing.** Beyond trusting the filename extension, the
  app now checks the first few bytes of an upload against common magic
  numbers (ZIP/OLE headers used by Excel files) and rejects a `.csv`/
  `.json` upload that's actually an Excel file in disguise, with a clear
  message, instead of letting it reach `pd.read_csv` and fail confusingly
  (or be silently misinterpreted).
- All three are covered by real tests, including two using Streamlit's
  official `AppTest` harness that actually run the app and simulate widget
  interactions (lock -> wrong password -> correct password -> unlocked).

CSV-export formula injection (a common Excel-related vulnerability class)
was checked and doesn't apply here: the app doesn't offer a raw-data CSV
download, only markdown/HTML/JSON reports.

## Tests

- `tests/test_edge_cases.py` (34 tests): hand-picked edge cases, now
  including regression tests for every Round 2 bug fix.
- `tests/test_fuzz.py` (9 property-based tests, ~50-150 examples each):
  randomized/malformed input across preprocessing, statistics,
  probability, visualization, and ML.
- `tests/test_visualizer.py` (16 tests): correctness + a real
  multi-threaded concurrency regression test for the OO matplotlib
  refactor.
- `tests/test_app_integration.py` (8 tests): real Streamlit `AppTest`
  integration tests for the access gate and upload sniffing.
- Original suite (41 tests) unchanged and still passing.

**Total: 110 tests, 0 skipped** (torch/transformers/shap were installed
into this environment specifically so DL/transformer/XAI paths run for
real rather than only exercising the graceful-`ImportError` fallback).

## CI

`.github/workflows/ci.yml` runs the full suite on Python 3.10/3.11/3.12,
once against core dependencies and again with `dl`/`xai`/`app` extras
installed, plus a non-blocking `ruff` lint pass.

## What's still true, even after this pass

- **Not every line of `classifier.py`/`clustering.py` was manually
  read.** The API sweep + fuzz testing is a systematic, evidence-based
  way to find *this class* of bug (unhandled exceptions escaping from
  degenerate inputs), and it found and fixed four real ones. It is not a
  formal proof of correctness for 4,600+ combined lines of code. There
  could be bugs in logic that *doesn't* raise an exception -- e.g. a
  metric computed incorrectly but which still returns a plausible-looking
  number -- which neither this sweep nor the fuzz suite would catch,
  since neither checks numerical correctness against a reference
  implementation.
- **No auth beyond a shared password**, no per-user accounts, no audit
  log of who did what. Fine for a small research team or classroom; not
  equivalent to an enterprise auth system.
- **Rate limiting and file-size caps are application-level, not
  infrastructure-level.** They protect against accidental overload (a
  confused user, a retry loop) far better than deliberate abuse. A
  motivated attacker with API access could still open many concurrent
  browser sessions, each under its own rate limit, and collectively
  overload the server -- that requires a reverse proxy, WAF, or
  infrastructure-level rate limiter/load balancer in front of this app,
  which is out of scope for application code.
- **Upload content-sniffing catches the common "renamed file" case**
  (ZIP/OLE magic bytes), not a general-purpose malware/content scanner.
  A file that's genuinely valid CSV but contains adversarial data (e.g.
  crafted to exploit a pandas/numpy vulnerability) would not be caught.
- **DL/transformer training paths were exercised for real this round**
  (torch/transformers/shap installed and all tests passing, none skipped)
  but only against small synthetic smoke-test data, not at any realistic
  training scale.
- **Fuzzing here checks for crashes, not correctness.** Hypothesis found
  no new unhandled-exception bugs at up to 150 examples/test in this pass,
  but that's evidence of absence within the tested input space, not proof
  there are none -- a longer/continuous fuzzing campaign (e.g. run as a
  nightly CI job with more examples) would raise confidence further.

## Round 3 -- mathematical correctness, ML leakage, reproducibility, security

This round worked through a 15-point external robustness review
(mathematical correctness, ML leakage, benchmark datasets, statistical
assumption checking, reproducibility, hostile-data testing, large-scale
performance, DL/NLP robustness, security hardening, code coverage,
fault injection, CI, research-report quality, and independent
validation). Below is exactly what was done, what was found, and --
per that review's own recommendation -- what is honestly still missing.
The headline claim this project can now support is **"tested against
reference values, checked for a specific class of ML leakage, and more
transparent about its own limits"** -- not "perfect," and not yet
"fully independently validated."

### Real bugs found and fixed

1. **App-wiring bug**: the Streamlit app called
   `InferentialStatistics.one_way_anova(group_col, value_col)`, but the
   method's signature is `one_way_anova(column, group_column)` -- the
   arguments were swapped, meaning every ANOVA run from the UI grouped
   by the wrong column entirely. Fixed in `app/main.py`.
2. **Real cross-validation leakage in `PsyRegressor.cross_validate()`**:
   it called `self._prepare(X, fit=True)` -- fitting `StandardScaler` on
   the *entire* dataset, across all folds -- before handing the scaled
   data to `cross_val_score()`. Every fold's "held-out" data had already
   influenced the scaler used to train on it, which is textbook
   preprocessing leakage and would make reported CV scores optimistic.
   Fixed by fitting the scaler *inside* each fold via an sklearn
   `Pipeline`. `tests/test_ml_benchmarks_and_leakage.py` proves the fix
   by monkey-patching `StandardScaler.fit` to record every call's sample
   size during 5-fold CV and asserting none of them ever equals the full
   dataset size.
3. **The same latent risk in `PsyClassifier.cross_validation()` /
   `stratified_cross_validation()`**: these didn't scale internally at
   all, so they weren't leaking by themselves, but calling them with
   data already scaled via `preprocess()` (a natural, undocumented usage
   pattern) would have leaked exactly like the regressor did. Fixed the
   same way -- scaling now happens inside a per-fold `Pipeline`
   (`scale=True` by default, matching prior behavior when callers scaled
   externally, but now safe either way).

### Mathematical correctness (`tests/test_statistical_correctness.py`, 19 tests)

Every descriptive/inferential statistic and Cronbach's alpha is now
checked against an **independently computed reference value** (hand
formulas evaluated via plain NumPy, not the code under test), not just
"does it run." Two non-bugs were caught and clarified by this process,
worth knowing about:

- `scipy.stats.chi2_contingency` applies the **Yates continuity
  correction by default for 2x2 tables** -- a naive hand-calculation
  without it will not match. The test now documents and applies the
  correction.
- **Cronbach's alpha equals 1 only for tau-equivalent items** (equal
  variance *and* perfect correlation), not merely perfectly *correlated*
  items with different scales/variances. This is correct psychometric
  behavior, not a bug, and is now an explicit regression test
  (`test_cronbach_alpha_of_rescaled_perfectly_correlated_items_is_below_one`)
  so nobody "fixes" it into a mathematically wrong shortcut later.

### Statistical assumption checking (item #4)

`InferentialStatistics` was substantially rewritten. Every test now
returns, alongside its statistic and p-value:

- **Sample size(s)** (`n`, `n1`/`n2`, `group_sizes`, etc.)
- **An effect size** -- Cohen's d (t-tests), eta-squared (ANOVA), or
  Cramer's V (chi-square) -- with a standard verbal interpretation
  (negligible/small/medium/large), since a p-value alone says nothing
  about practical significance.
- **A 95% confidence interval** where one has a standard closed form
  (means, mean differences via Welch-Satterthwaite df, Pearson/Spearman
  correlations via the Fisher z-transform).
- **Assumption diagnostics**: Shapiro-Wilk normality checks (skipped
  gracefully outside n=3..5000, or on constant data, rather than
  reporting a misleading result) and Levene's test for homogeneity of
  variance, each surfaced as a plain-English warning when violated (e.g.
  *"Levene's test suggests unequal variances between groups; Welch's
  t-test was used"*). `independent_t_test()` now defaults to
  auto-selecting Welch vs. Student's t-test based on Levene's result,
  rather than always forcing `equal_var=False` regardless of the data.
- **Input validation up front**: every method now raises a clear
  `ValueError` naming the problem (empty column, single-observation
  group, constant/zero-variance column, missing column, non-numeric
  data) instead of letting scipy/pandas raise a confusing exception, or
  worse, silently returning `nan`. Locked in by
  `TestInferentialStatisticsHostileData` in `tests/test_edge_cases.py`
  (18 new tests).

### Benchmark datasets (item #3, `tests/test_ml_benchmarks_and_leakage.py`)

`PsyClassifier.auto_train()` is checked against `sklearn`'s bundled
Iris and Breast Cancer Wisconsin datasets with a documented, deliberately
conservative accuracy floor (>=90%, well below what these easy,
well-studied datasets typically achieve). `PsyRegressor` is checked
against the Diabetes dataset with an R^2 floor of 0.25 for both
linear regression and random forest (a mean-only baseline gets R^2=0,
so this proves the model is learning real signal, not just running).
California Housing was deliberately **not** used because
`fetch_california_housing()` downloads from the internet on first call,
which won't work in a network-restricted CI/sandbox environment --
Diabetes (bundled with scikit-learn, no network needed) serves the same
purpose here.

### Reproducibility (item #5)

Added to `psyinsight/utils`:

- **`dataframe_fingerprint(df)`** -- a deterministic SHA-256 hash of a
  DataFrame's column names, dtypes, and row-level content (via
  `pandas.util.hash_pandas_object`). Identical data (including an exact
  copy) hashes identically; a value change, a column rename, a dtype
  change, or a row reorder all change the hash. This is the "dataset
  version" record the original review asked for.
- **`environment_fingerprint()`** -- Python version, platform string,
  and the installed version of every tracked package that's actually
  present (numpy/pandas/scipy/scikit-learn/matplotlib/joblib/streamlit/
  torch/transformers), captured via `importlib.metadata` with a
  `psyinsight` `__version__` fallback. Missing optional packages
  (torch/transformers) are silently omitted rather than raising.
- **`reproducibility_snapshot(df, seed, extra)`** -- bundles both of the
  above with the random seed and any caller-supplied metadata
  (hyperparameters, CV folds, etc.) into one dict.
- **`ResearchReportGenerator.add_reproducibility_section()`** -- one
  call attaches a full reproducibility snapshot to a research report, in
  both the Markdown and JSON export paths.

`set_seed()` (seeds python/numpy/torch) already existed from an earlier
pass and is unchanged; it's now paired with `reproducibility_snapshot()`
so the seed is actually *recorded* alongside a result, not just applied.

19 new tests in `tests/test_reproducibility.py` cover fingerprint
determinism (same data -> same hash, changed data/order/dtype -> a
different hash), environment capture, JSON-serializability, and report
integration.

**What reproducibility here does *not* guarantee**: some scikit-learn
estimators are not bit-for-bit deterministic under parallelism
(`n_jobs > 1`) even with a fixed seed, and this pass did not audit every
estimator's determinism under those conditions. The snapshot records
everything needed to *attempt* a reproduction; it doesn't prove one will
match to the last decimal.

### Security hardening (item #10)

Reviewed (not a full penetration test):

- **No hard-coded secrets** anywhere in `src/` or `app/` -- verified by
  a repo-wide grep for common secret-assignment patterns, now locked in
  by `TestNoHardcodedSecrets` so a future PR can't accidentally introduce
  one without a test noticing. The app's optional password gate reads
  from `st.secrets`, confirmed never a literal string.
- **`joblib.load()`** (used by every `save_model()`/`load()` pair in
  `classifier.py`, `regressor.py`, `clustering.py`) can execute arbitrary
  code, exactly like raw `pickle` -- this is a well-known, unavoidable
  property of joblib's format, not a bug to "fix" in this codebase. It
  is confirmed **not reachable from the Streamlit app today** (no
  `load_model(`/`PsyRegressor.load(` call sites in `app/main.py`,
  locked in by a test), and every `load_model`/`load` docstring now
  carries an explicit warning so this doesn't get wired up to an
  untrusted upload later without someone noticing the risk.
- **`DataLoader.load()`** takes a raw file path with no directory
  sandboxing. This is fine for its actual, current use (a trusted
  developer/script passing a local path, same trust model as `open()`)
  and is never exposed to end-users via the Streamlit app (which reads
  only from Streamlit's in-memory upload buffer). Documented as a
  "re-review before exposing" item rather than fixed, since sandboxing a
  path argument that legitimate callers need to point anywhere on disk
  would break its intended use.
- Upload size/row/column limits and content-sniffing (from Round 2) are
  unchanged and now explicitly locked in by
  `tests/test_security_and_fault_injection.py`.

**Explicitly not done** (documented, not silently skipped): dependency
vulnerability scanning (`pip-audit`/`safety` need to run as a CI step,
not a pytest test -- not added to `.github/workflows/ci.yml` in this
pass), secret-scanning as a pre-commit hook (a repo-config concern), and
audit logging of user actions. These are real gaps, not resolved by
anything in this round.

### Fault injection (item #12)

`tests/test_security_and_fault_injection.py::TestFaultInjection` covers:
a corrupted/truncated model file on load (raises, doesn't crash the
interpreter or return garbage), a missing model file, saving to a path
where a file blocks the expected parent directory (simulates a
disk/permissions failure), a missing CSV file, malformed JSON, and
`set_seed()`'s graceful behavior regardless of whether torch is
installed. This is a small, targeted set of the most plausible failure
modes for this application -- not a general chaos-engineering harness
(no simulated network partition, no simulated out-of-memory condition,
no simulated database failure, since this app has no database).

### Code coverage (item #11) -- measured honestly

```
pytest tests/ --cov=src/psyinsight --cov-report=term-missing
```

**Overall: 48% statement coverage** (up from not being measured at all).
This is well short of the reviewed target of 85-90%, and that gap should
not be hidden. Breakdown of where the coverage actually is and isn't:

| Area | Coverage | Note |
|---|---|---|
| `statistics/inferential.py` | 85% | Round 3 focus; well-covered |
| `preprocessing/*` | 69-100% | validator/cleaner fully covered |
| `regressor.py` | 83% | |
| `visualization/visualizer.py` | 99% | Round 2 focus |
| `classifier.py` | 45% | training/eval paths covered; most plotting/PDF-report code paths are not |
| `clustering.py` | 19% | the largest file (3,475 lines); only the core fit/predict/contingency paths used by tests are exercised -- the great majority of its plotting, PCA/t-SNE/UMAP visualization, and report-generation code is untested |
| `dl/*` | 15-21% | smoke-tested only (see Round 2 notes); most of the actual training loop internals are untested |
| `xai/explainer.py` | 62% | |

**Reading this honestly**: the modules this round focused on
(statistics, reproducibility, the leakage-fixed cross-validation paths)
are well-tested. The large ML/DL modules' *plotting and report
generation* code -- as opposed to their model-fitting/prediction logic,
which is exercised by the benchmark and fuzz tests -- is largely
untested. Getting `clustering.py` and `classifier.py`'s visualization
mixins to a respectable coverage number would be the single highest-
leverage next step for this metric specifically.

### Still not done (unchanged from the original review's list)

- **Large-scale performance testing** (1k/10k/100k/1M+ row runtime, RAM,
  concurrent-user benchmarking) -- not attempted this round. The app's
  own `MAX_ROWS = 200_000` guardrail limits worst-case input size, but
  no benchmark confirms performance is acceptable even at that ceiling.
- **DL/NLP robustness against adversarial/multi-lingual/emoji input** --
  not extended this round beyond what Round 2's fuzz suite already
  covers for the non-DL modules.
- **Automatic CI security/dependency scanning** -- not added to
  `.github/workflows/ci.yml`.
- **Independent validation by someone who didn't write the code** --
  by definition, cannot be done by an AI assistant working from the same
  codebase; this remains the single most valuable next step and the one
  most likely to surface issues that self-testing structurally cannot
  (e.g. a subtly wrong-but-plausible statistical interpretation, a UX
  assumption that doesn't hold for a real researcher's data).
- **Research-report overclaiming guardrails** (e.g. flagging
  correlation-as-causation language) -- `ResearchReportGenerator` still
  renders whatever content it's given; it doesn't yet check the
  *content* of a section for overclaiming language. Not addressed this
  round.

### Tests added this round

- `tests/test_statistical_correctness.py` -- 19 tests
- `tests/test_ml_benchmarks_and_leakage.py` -- 9 tests
- `tests/test_reproducibility.py` -- 19 tests
- `tests/test_security_and_fault_injection.py` -- 13 tests
- `tests/test_edge_cases.py` -- +18 tests (`TestInferentialStatisticsHostileData`)

**New total: 186 tests, 2 skipped, all passing.**
