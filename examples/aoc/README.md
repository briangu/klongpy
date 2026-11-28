# Advent of Code in Klong v2

The suite covers all 87 days of Advent of Code from 2022 through 2025. Each day is a strict-level-2 Klong module with one entry point. The 39 parts from the original Klong collection and 131 additional parts have checked answers. Every part links to its upstream program, fixture and expected-answer source. [Per-part source index and licenses](SOURCES.md)

| Year | Days | Answer-bearing parts | Examples and answers |
|---|---:|---:|---|
| 2022 | 25 | 49 | [2022 manifest](2022/cases.json) |
| 2023 | 25 | 49 | [2023 manifest](2023/cases.json) |
| 2024 | 25 | 49 | [2024 manifest](2024/cases.json) |
| 2025 | 12 | 23 | [2025 manifest](2025/cases.json) |

Each year's final day has one computational answer. The test inputs are public examples and project-authored fixtures. All can be run offline.

## Run a puzzle

From the repository root, build the CLI, check all 170 answers and ten generated workloads, or run one year.

```sh
cargo build --locked --release --manifest-path native/Cargo.toml -p klong-cli
python3 examples/aoc/benchmark.py --check
python3 examples/aoc/benchmark.py --workload sample --year 2024 --check
```

The driver takes the module, input file, part, number of timed batches, and one integer puzzle parameter. Zero batches runs the solver once. This example checks row 10 of the sensor map and prints a JSON object containing the answer 26.

```sh
native/target/release/kg --strict 2 examples/aoc/bench.kg \
  "$PWD/examples/aoc/2022/day15.kg" examples/aoc/samples/2022/15.txt 1 0 10
```

To run on the Python host, substitute `kgpy` and insert `--` after `bench.kg`. The test suite exercises every part on both desktop hosts and runs additional boundary cases.

```sh
.venv/bin/python -m pytest -q tests/v2/test_aoc*.py
```

Each module exports `solve(lines;part;parameter)`. Most days ignore the parameter. For these days it changes the problem being solved.

| Puzzle | Meaning | Example | Full puzzle |
|---|---|---|---|
| 2022 day 15 | Target row (part 1) or upper coordinate bound (part 2) | 10 or 20 | 2,000,000 or 4,000,000 |
| 2023 day 21 | Steps through the garden | 6 or 50 | 64 or 26,501,365 |
| 2023 day 24 part 1 | Test area | 7 selects 7–27 | 200000000000000 selects 200000000000000–400000000000000 |
| 2024 day 14 | Width of the torus | 11 selects 11×7 | 101 selects 101×103 |
| 2024 day 18 | Width of the board and initial byte count | 7 selects 12 bytes | 71 selects 1,024 bytes |
| 2024 day 20 | Minimum time saved by a shortcut | 2 or 50 | 100 |
| 2025 day 8 part 1 | Number of shortest connections to attempt | 10 | 1,000 |

For 2023 day 21 the solver extrapolates after five equal second differences at the target tile phase. It also matches the published 5,000-step example. The 2024 day 14 tree search recognizes at least 20 robots on one row with at least 11 consecutive occupied columns. Its synthetic fixture first meets that condition at tick 3. The 2024 day 17 quine search uses the puzzle program's octal loop and verifies its full output. For 2024 day 24 the repair assumes a ripple-carry adder and four disjoint output swaps. Symmetric circuits can admit multiple valid repairs. [Implementation and fixture sources](SOURCES.md)

## Measure a language change

Save a baseline, then run the same corpus with a second binary. Use `--year 2024 --day 22` to isolate the word-level XOR workload, for example. The harness checks the answer, input checksum and puzzle parameter before reporting a ratio. The JSON includes the runtime version and executable checksum, Klong source checksums, batch size, individual measurements and median for each case. A baseline/current ratio above one means the current solve was faster.

```sh
python3 examples/aoc/benchmark.py --workload scale --output before.json
python3 examples/aoc/benchmark.py --workload scale --kg /path/to/candidate/kg \
  --baseline before.json --output after.json
```

The warm benchmark times `solve` on input lines already in memory. It includes puzzle parsing and result construction. Each case gets a fresh process. The driver reads the fixture, times one warmup to choose a batch size targeting 10 ms, then checks the answers in each timed batch. Process startup, file I/O and JSON printing are outside the warm timing. The reported number is the median of seven per-solve batch means.

If the original `briangu/aoc` checkout is available, use `--classic ../aoc` to compare three original scripts with their ports on the **same executable**. The runner reads the pinned upstream scripts from Git and alternates launch order. These measurements include process startup, loading the source, I/O, the solve and printing the answer. Each script keeps its own entry point.

## Runtime improvements on the Apple M2 Ultra

All 170 puzzle parts and ten generated workloads returned the same expected answers before and after the runtime changes. Both native runs used seven timed batches per case on the same machine. The reports identify the binaries and Klong sources by SHA-256. [Original run](results/m2-ultra.json) · [Run with the new primitives](results/m2-ultra-primitives.json)

| Workload | Before | After | Before / after |
|---|---:|---:|---:|
| Hill Climbing, 1,051 cells, part 1 | 16.831 ms | 2.383 ms | 7.06× |
| Scratchcards, 4,096 cards, part 2 | 112.692 ms | 17.557 ms | 6.42× |
| Pulse Propagation, sample, part 1 | 24.921 ms | 25.412 ms | 0.98× |
| Hailstones, sample, part 2 | 0.236 ms | 0.155 ms | 1.52× |
| Chronospatial Computer, sample, part 2 | 2.895 ms | 2.872 ms | 1.01× |
| Monkey Market, four seeds, part 1 | 21.819 ms | 2.252 ms | 9.69× |
| Monkey Market, four seeds, part 2 | 30.022 ms | 14.462 ms | 2.08× |

Hill Climbing, Scratchcards and Pulse Propagation ran identical Klong sources in the two measurements. The first two benefit from reusing storage during indexed updates. Mixed-record append improved in isolation, while the full pulse simulation had little change on its sample. Three modules also changed source. The 2024 VM and market generator now call `.bitxor`, and market part 1 processes all seeds as a vector. The hailstone solver now uses exact integer operations and also passes a generated input whose correct answer exceeds the signed machine range. [Regression programs](../../tests/v2/programs) · [Integer primitives](../../docs/system-functions.md#integer-arithmetic)

## Apple M2 Ultra baseline

This pre-change run on November 27, 2026 used `kg 2.0.0-rc.1` on macOS arm64. All 180 cases matched their expected answers. The two module timings for 2022 day 19 and 2023 day 24 were refreshed after revisions to those solvers. [Raw measurements and checksums](results/m2-ultra.json)

| Generated workload | Input size | Part 1 warm median | Part 2 warm median |
|---|---|---:|---:|
| Rock Paper Scissors | 36,864 rounds | 28.20 ms | 28.41 ms |
| Tuning Trouble | 32,781 characters | 20.16 ms | 28.65 ms |
| Hill Climbing | 1,051 cells | 16.83 ms | 16.70 ms |
| Scratchcards | 4,096 cards | 12.38 ms | 112.69 ms |
| Seed mapping | one interval of 1 billion seeds | 12.8 µs | 12.5 µs |

The seed-mapping solver carries interval endpoints through the map, so interval width has little effect on this fixture. Scratchcards part 2 updates card multiplicities in a vector for every award. The corresponding part 1 timing helps isolate the cost of propagation from the shared parsing work.

| Fresh-process comparison | Original script | v2 module and driver |
|---|---:|---:|
| Rock Paper Scissors, part 1 | 88.64 ms | 56.93 ms |
| Tuning Trouble, four-character marker | 34.00 ms | 34.56 ms |
| Tuning Trouble, fourteen-character marker | 44.09 ms | 45.12 ms |

The scorer rewrite saved about 32 ms per fresh launch on the generated rounds. The two marker ports and the original scripts finished within about 1 ms of each other. Both columns used the same native v2 engine.

The [RC2 dogfooding notes](../../tests/v2/rc2-dogfooding.md#advent-of-code-language-findings) record the original failures, the fixes and the regression checks.
