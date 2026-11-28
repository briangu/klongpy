# Advent of Code sources

Each row identifies the upstream solution used for that Klong part, the exact input in this checkout, its origin, and the source of the expected answer. GitHub links pin a commit. The puzzle links credit Eric Wastl’s Advent of Code. The examples run offline.

The original Klong programs are Brian Guarraci’s. The Python references are Marcus Read’s MIT-licensed solutions. The independent Rust test vectors and assertions come from maneatingape. Both upstream MIT notices are included verbatim. [Original Klong tree](https://github.com/briangu/aoc/tree/defeb2e63ade7aed486936e5de18dfb7f2db7907) · [Python license](references/maread99-LICENSE.txt) · [Rust test license](references/maneatingape-LICENSE.txt)

The notice copies match their upstream files byte for byte. [Original Python notice](https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/LICENSE.txt) · [Original Rust test notice](https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/LICENSE.md)

The last column identifies a published example, an assertion, an upstream program run on the local fixture, or an expected value derived for a project-authored case. All expected values and run parameters are in the linked year manifests. The ten larger workloads and their expected values were developed in this project. [Fixture generator](generate.kg) · [Scale answer calculations](benchmark.py)


## 2022

[Expected answers and parameters](2022/cases.json)

| Klong part | Upstream solution | Local input | Input origin | Expected answer source |
|---|---|---|---|---|
| [2022/01/1](2022/day01.kg) | [1.kg][kg-2022-1] | [1.txt](samples/2022/1.txt) | [puzzle example][puzzle-2022-1] | [Rust test][rust-2022-1] |
| [2022/01/2](2022/day01.kg) | [1b.kg][kg-2022-1b] | [1.txt](samples/2022/1.txt) | [puzzle example][puzzle-2022-1] | [Rust test][rust-2022-1] |
| [2022/02/1](2022/day02.kg) | [2.kg][kg-2022-2] | [2.txt](samples/2022/2.txt) | [puzzle example][puzzle-2022-2] | [Rust test][rust-2022-2] |
| [2022/02/2](2022/day02.kg) | [2b.kg][kg-2022-2b] | [2.txt](samples/2022/2.txt) | [puzzle example][puzzle-2022-2] | [Rust test][rust-2022-2] |
| [2022/03/1](2022/day03.kg) | [3.kg][kg-2022-3] | [3.txt](samples/2022/3.txt) | [puzzle example][puzzle-2022-3] | [Rust test][rust-2022-3] |
| [2022/03/2](2022/day03.kg) | [3b.kg][kg-2022-3b] | [3.txt](samples/2022/3.txt) | [puzzle example][puzzle-2022-3] | [Rust test][rust-2022-3] |
| [2022/04/1](2022/day04.kg) | [4.kg][kg-2022-4] | [4.txt](samples/2022/4.txt) | [puzzle example][puzzle-2022-4] | [Rust test][rust-2022-4] |
| [2022/04/2](2022/day04.kg) | [4b.kg][kg-2022-4b] | [4.txt](samples/2022/4.txt) | [puzzle example][puzzle-2022-4] | [Rust test][rust-2022-4] |
| [2022/05/1](2022/day05.kg) | [5.kg][kg-2022-5] | [5.txt](samples/2022/5.txt) | [5.test][input-2022-5.test] | [Rust test][rust-2022-5] |
| [2022/05/2](2022/day05.kg) | [5b.kg][kg-2022-5b] | [5.txt](samples/2022/5.txt) | [5.test][input-2022-5.test] | [Rust test][rust-2022-5] |
| [2022/06/1](2022/day06.kg) | [6.kg][kg-2022-6] | [6.txt](samples/2022/6.txt) | [puzzle example][puzzle-2022-6] | [Rust test][rust-2022-6] |
| [2022/06/2](2022/day06.kg) | [6b.kg][kg-2022-6b] | [6.txt](samples/2022/6.txt) | [puzzle example][puzzle-2022-6] | [Rust test][rust-2022-6] |
| [2022/07/1](2022/day07.kg) | [7.kg][kg-2022-7] | [7.txt](samples/2022/7.txt) | [7.test][input-2022-7.test] | [Rust test][rust-2022-7] |
| [2022/07/2](2022/day07.kg) | [7b.kg][kg-2022-7b] | [7.txt](samples/2022/7.txt) | [7.test][input-2022-7.test] | [Rust test][rust-2022-7] |
| [2022/08/1](2022/day08.kg) | [8.kg][kg-2022-8] | [8.txt](samples/2022/8.txt) | [8.test][input-2022-8.test] | [Rust test][rust-2022-8] |
| [2022/08/2](2022/day08.kg) | [8b.kg][kg-2022-8b] | [8.txt](samples/2022/8.txt) | [8.test][input-2022-8.test] | [Rust test][rust-2022-8] |
| [2022/09/1](2022/day09.kg) | [9.kg][kg-2022-9] | [9.txt](samples/2022/9.txt) | [9.test][input-2022-9.test] | [Rust test][rust-2022-9] |
| [2022/09/2](2022/day09.kg) | [9b.kg][kg-2022-9b] | [9b.txt](samples/2022/9b.txt) | [9b.test][input-2022-9b.test] | [puzzle example][puzzle-2022-9] |
| [2022/10/1](2022/day10.kg) | [10.kg][kg-2022-10] | [10.txt](samples/2022/10.txt) | [10.test][input-2022-10.test] | [Rust test][rust-2022-10] |
| [2022/10/2](2022/day10.kg) | [10b.kg][kg-2022-10b] | [10.txt](samples/2022/10.txt) | [10.test][input-2022-10.test] | [Rust test][rust-2022-10] |
| [2022/11/1](2022/day11.kg) | [11.kg][kg-2022-11] | [11.txt](samples/2022/11.txt) | [11.test][input-2022-11.test] | [Rust test][rust-2022-11] |
| [2022/11/2](2022/day11.kg) | [11b.kg][kg-2022-11b] | [11.txt](samples/2022/11.txt) | [11.test][input-2022-11.test] | [Rust test][rust-2022-11] |
| [2022/12/1](2022/day12.kg) | [12.kg][kg-2022-12] | [12.txt](samples/2022/12.txt) | [12.test][input-2022-12.test] | [Rust test][rust-2022-12] |
| [2022/12/2](2022/day12.kg) | [12b.kg][kg-2022-12b] | [12.txt](samples/2022/12.txt) | [12.test][input-2022-12.test] | [Rust test][rust-2022-12] |
| [2022/13/1](2022/day13.kg) | [13.kg][kg-2022-13] | [13.txt](samples/2022/13.txt) | [13.test][input-2022-13.test] | [Rust test][rust-2022-13] |
| [2022/13/2](2022/day13.kg) | [13b.kg][kg-2022-13b] | [13.txt](samples/2022/13.txt) | [13.test][input-2022-13.test] | [Rust test][rust-2022-13] |
| [2022/14/1](2022/day14.kg) | [14.kg][kg-2022-14] | [14.txt](samples/2022/14.txt) | [14.test][input-2022-14.test] | [Rust test][rust-2022-14] |
| [2022/14/2](2022/day14.kg) | [14b.kg][kg-2022-14b] | [14.txt](samples/2022/14.txt) | [14.test][input-2022-14.test] | [Rust test][rust-2022-14] |
| [2022/15/1](2022/day15.kg) | [15.kg][kg-2022-15] (unfinished), [day15.py][python-2022-day15.py] | [15.txt](samples/2022/15.txt) | [15.test][input-2022-15.test] | [Rust test][rust-2022-15] |
| [2022/15/2](2022/day15.kg) | [day15.py][python-2022-day15.py] | [15.txt](samples/2022/15.txt) | [15.test][input-2022-15.test] | [Rust test][rust-2022-15] |
| [2022/16/1](2022/day16.kg) | [day16_rev2.py][python-2022-day16_rev2.py] | [16.txt](samples/2022/16.txt) | [Rust test][rust-2022-16] | [Rust test][rust-2022-16] |
| [2022/16/2](2022/day16.kg) | [day16_rev2.py][python-2022-day16_rev2.py] | [16.txt](samples/2022/16.txt) | [Rust test][rust-2022-16] | [Rust test][rust-2022-16] |
| [2022/17/1](2022/day17.kg) | [day17.py][python-2022-day17.py] | [17.txt](samples/2022/17.txt) | [Rust test][rust-2022-17] | [Rust test][rust-2022-17] |
| [2022/17/2](2022/day17.kg) | [day17.py][python-2022-day17.py] | [17.txt](samples/2022/17.txt) | [Rust test][rust-2022-17] | [Rust test][rust-2022-17] |
| [2022/18/1](2022/day18.kg) | [day18.py][python-2022-day18.py] | [18.txt](samples/2022/18.txt) | [Rust test][rust-2022-18] | [Rust test][rust-2022-18] |
| [2022/18/2](2022/day18.kg) | [day18.py][python-2022-day18.py] | [18.txt](samples/2022/18.txt) | [Rust test][rust-2022-18] | [Rust test][rust-2022-18] |
| [2022/19/1](2022/day19.kg) | [day19.py][python-2022-day19.py] | [19.txt](samples/2022/19.txt) | [Rust test][rust-2022-19] | [Rust test][rust-2022-19] |
| [2022/19/2](2022/day19.kg) | [day19.py][python-2022-day19.py] | [19.txt](samples/2022/19.txt) | [Rust test][rust-2022-19] | [Rust test][rust-2022-19] |
| [2022/20/1](2022/day20.kg) | [day20.py][python-2022-day20.py] | [20.txt](samples/2022/20.txt) | [Rust test][rust-2022-20] | [Rust test][rust-2022-20] |
| [2022/20/2](2022/day20.kg) | [day20.py][python-2022-day20.py] | [20.txt](samples/2022/20.txt) | [Rust test][rust-2022-20] | [Rust test][rust-2022-20] |
| [2022/21/1](2022/day21.kg) | [day21.py][python-2022-day21.py] | [21.txt](samples/2022/21.txt) | [Rust test][rust-2022-21] | [Rust test][rust-2022-21] |
| [2022/21/2](2022/day21.kg) | [day21.py][python-2022-day21.py] | [21.txt](samples/2022/21.txt) | [Rust test][rust-2022-21] | [Rust test][rust-2022-21] |
| [2022/22/1](2022/day22.kg) | [day22.py][python-2022-day22.py] | [22.txt](samples/2022/22.txt) | [Rust test][rust-2022-22] | [Rust test][rust-2022-22] |
| [2022/22/2](2022/day22.kg) | [day22.py][python-2022-day22.py] | [22.txt](samples/2022/22.txt) | [Rust test][rust-2022-22] | [Rust test][rust-2022-22] |
| [2022/23/1](2022/day23.kg) | [day23.py][python-2022-day23.py] | [23.txt](samples/2022/23.txt) | [Rust test][rust-2022-23] | [Rust test][rust-2022-23] |
| [2022/23/2](2022/day23.kg) | [day23.py][python-2022-day23.py] | [23.txt](samples/2022/23.txt) | [Rust test][rust-2022-23] | [Rust test][rust-2022-23] |
| [2022/24/1](2022/day24.kg) | [day24.py][python-2022-day24.py] | [24.txt](samples/2022/24.txt) | [Rust test][rust-2022-24] | [Rust test][rust-2022-24] |
| [2022/24/2](2022/day24.kg) | [day24.py][python-2022-day24.py] | [24.txt](samples/2022/24.txt) | [Rust test][rust-2022-24] | [Rust test][rust-2022-24] |
| [2022/25/1](2022/day25.kg) | [day25.py][python-2022-day25.py] | [25.txt](samples/2022/25.txt) | [Rust test][rust-2022-25] | [Rust test][rust-2022-25] |

Puzzle days · [01][puzzle-2022-1] · [02][puzzle-2022-2] · [03][puzzle-2022-3] · [04][puzzle-2022-4] · [05][puzzle-2022-5] · [06][puzzle-2022-6] · [07][puzzle-2022-7] · [08][puzzle-2022-8] · [09][puzzle-2022-9] · [10][puzzle-2022-10] · [11][puzzle-2022-11] · [12][puzzle-2022-12] · [13][puzzle-2022-13] · [14][puzzle-2022-14] · [15][puzzle-2022-15] · [16][puzzle-2022-16] · [17][puzzle-2022-17] · [18][puzzle-2022-18] · [19][puzzle-2022-19] · [20][puzzle-2022-20] · [21][puzzle-2022-21] · [22][puzzle-2022-22] · [23][puzzle-2022-23] · [24][puzzle-2022-24] · [25][puzzle-2022-25]

[kg-2022-1]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/1.kg
[puzzle-2022-1]: https://adventofcode.com/2022/day/1
[rust-2022-1]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day01.rs
[kg-2022-1b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/1b.kg
[kg-2022-2]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/2.kg
[puzzle-2022-2]: https://adventofcode.com/2022/day/2
[rust-2022-2]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day02.rs
[kg-2022-2b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/2b.kg
[kg-2022-3]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/3.kg
[puzzle-2022-3]: https://adventofcode.com/2022/day/3
[rust-2022-3]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day03.rs
[kg-2022-3b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/3b.kg
[kg-2022-4]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/4.kg
[puzzle-2022-4]: https://adventofcode.com/2022/day/4
[rust-2022-4]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day04.rs
[kg-2022-4b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/4b.kg
[kg-2022-5]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/5.kg
[puzzle-2022-5]: https://adventofcode.com/2022/day/5
[rust-2022-5]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day05.rs
[input-2022-5.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/5.test
[kg-2022-5b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/5b.kg
[kg-2022-6]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/6.kg
[puzzle-2022-6]: https://adventofcode.com/2022/day/6
[rust-2022-6]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day06.rs
[kg-2022-6b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/6b.kg
[kg-2022-7]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/7.kg
[puzzle-2022-7]: https://adventofcode.com/2022/day/7
[rust-2022-7]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day07.rs
[input-2022-7.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/7.test
[kg-2022-7b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/7b.kg
[kg-2022-8]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/8.kg
[puzzle-2022-8]: https://adventofcode.com/2022/day/8
[rust-2022-8]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day08.rs
[input-2022-8.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/8.test
[kg-2022-8b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/8b.kg
[kg-2022-9]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/9.kg
[puzzle-2022-9]: https://adventofcode.com/2022/day/9
[rust-2022-9]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day09.rs
[input-2022-9.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/9.test
[kg-2022-9b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/9b.kg
[input-2022-9b.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/9b.test
[kg-2022-10]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/10.kg
[puzzle-2022-10]: https://adventofcode.com/2022/day/10
[rust-2022-10]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day10.rs
[input-2022-10.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/10.test
[kg-2022-10b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/10b.kg
[kg-2022-11]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/11.kg
[puzzle-2022-11]: https://adventofcode.com/2022/day/11
[rust-2022-11]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day11.rs
[input-2022-11.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/11.test
[kg-2022-11b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/11b.kg
[kg-2022-12]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/12.kg
[puzzle-2022-12]: https://adventofcode.com/2022/day/12
[rust-2022-12]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day12.rs
[input-2022-12.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/12.test
[kg-2022-12b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/12b.kg
[kg-2022-13]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/13.kg
[puzzle-2022-13]: https://adventofcode.com/2022/day/13
[rust-2022-13]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day13.rs
[input-2022-13.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/13.test
[kg-2022-13b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/13b.kg
[kg-2022-14]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/14.kg
[puzzle-2022-14]: https://adventofcode.com/2022/day/14
[rust-2022-14]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day14.rs
[input-2022-14.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/14.test
[kg-2022-14b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/14b.kg
[kg-2022-15]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/15.kg
[python-2022-day15.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day15.py
[puzzle-2022-15]: https://adventofcode.com/2022/day/15
[rust-2022-15]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day15.rs
[input-2022-15.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/15.test
[python-2022-day16_rev2.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day16_rev2.py
[puzzle-2022-16]: https://adventofcode.com/2022/day/16
[rust-2022-16]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day16.rs
[python-2022-day17.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day17.py
[puzzle-2022-17]: https://adventofcode.com/2022/day/17
[rust-2022-17]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day17.rs
[python-2022-day18.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day18.py
[puzzle-2022-18]: https://adventofcode.com/2022/day/18
[rust-2022-18]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day18.rs
[python-2022-day19.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day19.py
[puzzle-2022-19]: https://adventofcode.com/2022/day/19
[rust-2022-19]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day19.rs
[python-2022-day20.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day20.py
[puzzle-2022-20]: https://adventofcode.com/2022/day/20
[rust-2022-20]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day20.rs
[python-2022-day21.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day21.py
[puzzle-2022-21]: https://adventofcode.com/2022/day/21
[rust-2022-21]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day21.rs
[python-2022-day22.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day22.py
[puzzle-2022-22]: https://adventofcode.com/2022/day/22
[rust-2022-22]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day22.rs
[python-2022-day23.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day23.py
[puzzle-2022-23]: https://adventofcode.com/2022/day/23
[rust-2022-23]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day23.rs
[python-2022-day24.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day24.py
[puzzle-2022-24]: https://adventofcode.com/2022/day/24
[rust-2022-24]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day24.rs
[python-2022-day25.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2022/day25.py
[puzzle-2022-25]: https://adventofcode.com/2022/day/25
[rust-2022-25]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2022/day25.rs

## 2023

[Expected answers and parameters](2023/cases.json)

| Klong part | Upstream solution | Local input | Input origin | Expected answer source |
|---|---|---|---|---|
| [2023/01/1](2023/day01.kg) | [1.kg][kg-2023-1] | [1.txt](samples/2023/1.txt) | [1.test][input-2023-1.test] | [Rust test][rust-2023-1] |
| [2023/01/2](2023/day01.kg) | [1b.kg][kg-2023-1b] | [1b.txt](samples/2023/1b.txt) | [1b.test][input-2023-1b.test] | [Rust test][rust-2023-1] |
| [2023/02/1](2023/day02.kg) | [2.kg][kg-2023-2] | [2.txt](samples/2023/2.txt) | [2.test][input-2023-2.test] | [Rust test][rust-2023-2] |
| [2023/02/2](2023/day02.kg) | [2b.kg][kg-2023-2b] | [2.txt](samples/2023/2.txt) | [2.test][input-2023-2.test] | [Rust test][rust-2023-2] |
| [2023/03/1](2023/day03.kg) | [3.kg][kg-2023-3] | [3.txt](samples/2023/3.txt) | [3.test][input-2023-3.test] | [Rust test][rust-2023-3] |
| [2023/03/2](2023/day03.kg) | [3b.kg][kg-2023-3b] | [3.txt](samples/2023/3.txt) | [3.test][input-2023-3.test] | [Rust test][rust-2023-3] |
| [2023/04/1](2023/day04.kg) | [4.kg][kg-2023-4] | [4.txt](samples/2023/4.txt) | [4.test][input-2023-4.test] | [Rust test][rust-2023-4] |
| [2023/04/2](2023/day04.kg) | [4b.kg][kg-2023-4b] | [4.txt](samples/2023/4.txt) | [4.test][input-2023-4.test] | [Rust test][rust-2023-4] |
| [2023/05/1](2023/day05.kg) | [5.kg][kg-2023-5] | [5.txt](samples/2023/5.txt) | [5.test][input-2023-5.test] | [Rust test][rust-2023-5] |
| [2023/05/2](2023/day05.kg) | [5b.kg][kg-2023-5b] | [5.txt](samples/2023/5.txt) | [5.test][input-2023-5.test] | [Rust test][rust-2023-5] |
| [2023/06/1](2023/day06.kg) | [06.py][python-2023-06.py] | [6.txt](samples/2023/6.txt) | [Rust test][rust-2023-6] | [Rust test][rust-2023-6] |
| [2023/06/2](2023/day06.kg) | [06.py][python-2023-06.py] | [6.txt](samples/2023/6.txt) | [Rust test][rust-2023-6] | [Rust test][rust-2023-6] |
| [2023/07/1](2023/day07.kg) | [07.py][python-2023-07.py] | [7.txt](samples/2023/7.txt) | [Rust test][rust-2023-7] | [Rust test][rust-2023-7] |
| [2023/07/2](2023/day07.kg) | [07.py][python-2023-07.py] | [7.txt](samples/2023/7.txt) | [Rust test][rust-2023-7] | [Rust test][rust-2023-7] |
| [2023/08/1](2023/day08.kg) | [08.py][python-2023-08.py] | [8.txt](samples/2023/8.txt) | [Rust test][rust-2023-8] | [Rust test][rust-2023-8] |
| [2023/08/2](2023/day08.kg) | [08.py][python-2023-08.py] | [8b.txt](samples/2023/8b.txt) | [Rust test][rust-2023-8] | [Rust test][rust-2023-8] |
| [2023/09/1](2023/day09.kg) | [09.py][python-2023-09.py] | [9.txt](samples/2023/9.txt) | [Rust test][rust-2023-9] | [Rust test][rust-2023-9] |
| [2023/09/2](2023/day09.kg) | [09.py][python-2023-09.py] | [9.txt](samples/2023/9.txt) | [Rust test][rust-2023-9] | [Rust test][rust-2023-9] |
| [2023/10/1](2023/day10.kg) | [10.py][python-2023-10.py] | [10.txt](samples/2023/10.txt) | [Rust test][rust-2023-10] | [Rust test][rust-2023-10] |
| [2023/10/2](2023/day10.kg) | [10.py][python-2023-10.py] | [10b.txt](samples/2023/10b.txt) | [Rust test][rust-2023-10] | [Rust test][rust-2023-10] |
| [2023/11/1](2023/day11.kg) | [11.py][python-2023-11.py] | [11.txt](samples/2023/11.txt) | [Rust test][rust-2023-11] | [Rust test][rust-2023-11] |
| [2023/11/2](2023/day11.kg) | [11.py][python-2023-11.py] | [11.txt](samples/2023/11.txt) | [Rust test][rust-2023-11] | [Rust test][rust-2023-11] |
| [2023/12/1](2023/day12.kg) | [12.py][python-2023-12.py] | [12.txt](samples/2023/12.txt) | [Rust test][rust-2023-12] | [Rust test][rust-2023-12] |
| [2023/12/2](2023/day12.kg) | [12.py][python-2023-12.py] | [12.txt](samples/2023/12.txt) | [Rust test][rust-2023-12] | [Rust test][rust-2023-12] |
| [2023/13/1](2023/day13.kg) | [13.py][python-2023-13.py] | [13.txt](samples/2023/13.txt) | [Rust test][rust-2023-13] | [Rust test][rust-2023-13] |
| [2023/13/2](2023/day13.kg) | [13.py][python-2023-13.py] | [13.txt](samples/2023/13.txt) | [Rust test][rust-2023-13] | [Rust test][rust-2023-13] |
| [2023/14/1](2023/day14.kg) | [14.py][python-2023-14.py] | [14.txt](samples/2023/14.txt) | [Rust test][rust-2023-14] | [Rust test][rust-2023-14] |
| [2023/14/2](2023/day14.kg) | [14.py][python-2023-14.py] | [14.txt](samples/2023/14.txt) | [Rust test][rust-2023-14] | [Rust test][rust-2023-14] |
| [2023/15/1](2023/day15.kg) | [15.py][python-2023-15.py] | [15.txt](samples/2023/15.txt) | [Rust test][rust-2023-15] | [Rust test][rust-2023-15] |
| [2023/15/2](2023/day15.kg) | [15.py][python-2023-15.py] | [15.txt](samples/2023/15.txt) | [Rust test][rust-2023-15] | [Rust test][rust-2023-15] |
| [2023/16/1](2023/day16.kg) | [16.py][python-2023-16.py] | [16.txt](samples/2023/16.txt) | [Rust test][rust-2023-16] | [Rust test][rust-2023-16] |
| [2023/16/2](2023/day16.kg) | [16.py][python-2023-16.py] | [16.txt](samples/2023/16.txt) | [Rust test][rust-2023-16] | [Rust test][rust-2023-16] |
| [2023/17/1](2023/day17.kg) | [17.py][python-2023-17.py] | [17.txt](samples/2023/17.txt) | [Rust test][rust-2023-17] | [Rust test][rust-2023-17] |
| [2023/17/2](2023/day17.kg) | [17.py][python-2023-17.py] | [17.txt](samples/2023/17.txt) | [Rust test][rust-2023-17] | [Rust test][rust-2023-17] |
| [2023/18/1](2023/day18.kg) | [18.py][python-2023-18.py] | [18.txt](samples/2023/18.txt) | [Rust test][rust-2023-18] | [Rust test][rust-2023-18] |
| [2023/18/2](2023/day18.kg) | [18.py][python-2023-18.py] | [18.txt](samples/2023/18.txt) | [Rust test][rust-2023-18] | [Rust test][rust-2023-18] |
| [2023/19/1](2023/day19.kg) | [19.py][python-2023-19.py] | [19.txt](samples/2023/19.txt) | [Rust test][rust-2023-19] | [Rust test][rust-2023-19] |
| [2023/19/2](2023/day19.kg) | [19.py][python-2023-19.py] | [19.txt](samples/2023/19.txt) | [Rust test][rust-2023-19] | [Rust test][rust-2023-19] |
| [2023/20/1](2023/day20.kg) | [20.py][python-2023-20.py] | [20.txt](samples/2023/20.txt) | [20.py][python-2023-20.py] | [puzzle example][puzzle-2023-20] |
| [2023/20/2](2023/day20.kg) | [20.py][python-2023-20.py] | [20b.txt](samples/2023/20b.txt) | Project-authored pulse network | Independent FIFO simulation, first low pulse at press 15 ([case record](2023/cases.json)) |
| [2023/21/1](2023/day21.kg) | [21_a.py][python-2023-21_a.py] | [21.txt](samples/2023/21.txt) | [21_a.py][python-2023-21_a.py] | [puzzle example][puzzle-2023-21] |
| [2023/21/2](2023/day21.kg) | [21_b.py][python-2023-21_b.py] | [21.txt](samples/2023/21.txt) | [21_a.py][python-2023-21_a.py] | [puzzle example][puzzle-2023-21] |
| [2023/22/1](2023/day22.kg) | [22.py][python-2023-22.py] | [22.txt](samples/2023/22.txt) | [Rust test][rust-2023-22] | [Rust test][rust-2023-22] |
| [2023/22/2](2023/day22.kg) | [22.py][python-2023-22.py] | [22.txt](samples/2023/22.txt) | [Rust test][rust-2023-22] | [Rust test][rust-2023-22] |
| [2023/23/1](2023/day23.kg) | [23_a.py][python-2023-23_a.py] | [23.txt](samples/2023/23.txt) | [23_a.py][python-2023-23_a.py] | [puzzle example][puzzle-2023-23] |
| [2023/23/2](2023/day23.kg) | [23_b.py][python-2023-23_b.py] | [23.txt](samples/2023/23.txt) | [23_a.py][python-2023-23_a.py] | [puzzle example][puzzle-2023-23] |
| [2023/24/1](2023/day24.kg) | [24.py][python-2023-24.py] | [24.txt](samples/2023/24.txt) | [Rust test][rust-2023-24] | [puzzle example][puzzle-2023-24] |
| [2023/24/2](2023/day24.kg) | [24.py][python-2023-24.py] | [24.txt](samples/2023/24.txt) | [Rust test][rust-2023-24] | [Rust test][rust-2023-24] |
| [2023/25/1](2023/day25.kg) | [25.py][python-2023-25.py] | [25.txt](samples/2023/25.txt) | [Rust test][rust-2023-25] | [Rust test][rust-2023-25] |

Puzzle days · [01][puzzle-2023-1] · [02][puzzle-2023-2] · [03][puzzle-2023-3] · [04][puzzle-2023-4] · [05][puzzle-2023-5] · [06][puzzle-2023-6] · [07][puzzle-2023-7] · [08][puzzle-2023-8] · [09][puzzle-2023-9] · [10][puzzle-2023-10] · [11][puzzle-2023-11] · [12][puzzle-2023-12] · [13][puzzle-2023-13] · [14][puzzle-2023-14] · [15][puzzle-2023-15] · [16][puzzle-2023-16] · [17][puzzle-2023-17] · [18][puzzle-2023-18] · [19][puzzle-2023-19] · [20][puzzle-2023-20] · [21][puzzle-2023-21] · [22][puzzle-2023-22] · [23][puzzle-2023-23] · [24][puzzle-2023-24] · [25][puzzle-2023-25]

[kg-2023-1]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/1.kg
[puzzle-2023-1]: https://adventofcode.com/2023/day/1
[rust-2023-1]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day01.rs
[input-2023-1.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/1.test
[kg-2023-1b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/1b.kg
[input-2023-1b.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/1b.test
[kg-2023-2]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/2.kg
[puzzle-2023-2]: https://adventofcode.com/2023/day/2
[rust-2023-2]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day02.rs
[input-2023-2.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/2.test
[kg-2023-2b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/2b.kg
[kg-2023-3]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/3.kg
[puzzle-2023-3]: https://adventofcode.com/2023/day/3
[rust-2023-3]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day03.rs
[input-2023-3.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/3.test
[kg-2023-3b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/3b.kg
[kg-2023-4]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/4.kg
[puzzle-2023-4]: https://adventofcode.com/2023/day/4
[rust-2023-4]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day04.rs
[input-2023-4.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/4.test
[kg-2023-4b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/4b.kg
[kg-2023-5]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/5.kg
[puzzle-2023-5]: https://adventofcode.com/2023/day/5
[rust-2023-5]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day05.rs
[input-2023-5.test]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/5.test
[kg-2023-5b]: https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/5b.kg
[python-2023-06.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/06.py
[puzzle-2023-6]: https://adventofcode.com/2023/day/6
[rust-2023-6]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day06.rs
[python-2023-07.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/07.py
[puzzle-2023-7]: https://adventofcode.com/2023/day/7
[rust-2023-7]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day07.rs
[python-2023-08.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/08.py
[puzzle-2023-8]: https://adventofcode.com/2023/day/8
[rust-2023-8]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day08.rs
[python-2023-09.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/09.py
[puzzle-2023-9]: https://adventofcode.com/2023/day/9
[rust-2023-9]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day09.rs
[python-2023-10.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/10.py
[puzzle-2023-10]: https://adventofcode.com/2023/day/10
[rust-2023-10]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day10.rs
[python-2023-11.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/11.py
[puzzle-2023-11]: https://adventofcode.com/2023/day/11
[rust-2023-11]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day11.rs
[python-2023-12.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/12.py
[puzzle-2023-12]: https://adventofcode.com/2023/day/12
[rust-2023-12]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day12.rs
[python-2023-13.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/13.py
[puzzle-2023-13]: https://adventofcode.com/2023/day/13
[rust-2023-13]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day13.rs
[python-2023-14.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/14.py
[puzzle-2023-14]: https://adventofcode.com/2023/day/14
[rust-2023-14]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day14.rs
[python-2023-15.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/15.py
[puzzle-2023-15]: https://adventofcode.com/2023/day/15
[rust-2023-15]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day15.rs
[python-2023-16.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/16.py
[puzzle-2023-16]: https://adventofcode.com/2023/day/16
[rust-2023-16]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day16.rs
[python-2023-17.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/17.py
[puzzle-2023-17]: https://adventofcode.com/2023/day/17
[rust-2023-17]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day17.rs
[python-2023-18.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/18.py
[puzzle-2023-18]: https://adventofcode.com/2023/day/18
[rust-2023-18]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day18.rs
[python-2023-19.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/19.py
[puzzle-2023-19]: https://adventofcode.com/2023/day/19
[rust-2023-19]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day19.rs
[python-2023-20.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/20.py
[puzzle-2023-20]: https://adventofcode.com/2023/day/20
[python-2023-21_a.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/21_a.py
[puzzle-2023-21]: https://adventofcode.com/2023/day/21
[python-2023-21_b.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/21_b.py
[python-2023-22.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/22.py
[puzzle-2023-22]: https://adventofcode.com/2023/day/22
[rust-2023-22]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day22.rs
[python-2023-23_a.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/23_a.py
[puzzle-2023-23]: https://adventofcode.com/2023/day/23
[python-2023-23_b.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/23_b.py
[python-2023-24.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/24.py
[puzzle-2023-24]: https://adventofcode.com/2023/day/24
[rust-2023-24]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day24.rs
[python-2023-25.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2023/25.py
[puzzle-2023-25]: https://adventofcode.com/2023/day/25
[rust-2023-25]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2023/day25.rs

## 2024

[Expected answers and parameters](2024/cases.json)

| Klong part | Upstream solution | Local input | Input origin | Expected answer source |
|---|---|---|---|---|
| [2024/01/1](2024/day01.kg) | [01.py][python-2024-01.py] | [1.txt](samples/2024/1.txt) | [Rust test][rust-2024-1] | [Rust test][rust-2024-1] |
| [2024/01/2](2024/day01.kg) | [01.py][python-2024-01.py] | [1.txt](samples/2024/1.txt) | [Rust test][rust-2024-1] | [Rust test][rust-2024-1] |
| [2024/02/1](2024/day02.kg) | [02.py][python-2024-02.py] | [2.txt](samples/2024/2.txt) | [Rust test][rust-2024-2] | [Rust test][rust-2024-2] |
| [2024/02/2](2024/day02.kg) | [02.py][python-2024-02.py] | [2.txt](samples/2024/2.txt) | [Rust test][rust-2024-2] | [Rust test][rust-2024-2] |
| [2024/03/1](2024/day03.kg) | [03.py][python-2024-03.py] | [3.txt](samples/2024/3.txt) | [Rust test][rust-2024-3] | [Rust test][rust-2024-3] |
| [2024/03/2](2024/day03.kg) | [03.py][python-2024-03.py] | [3b.txt](samples/2024/3b.txt) | [Rust test][rust-2024-3] | [Rust test][rust-2024-3] |
| [2024/04/1](2024/day04.kg) | [04.py][python-2024-04.py] | [4.txt](samples/2024/4.txt) | [Rust test][rust-2024-4] | [Rust test][rust-2024-4] |
| [2024/04/2](2024/day04.kg) | [04.py][python-2024-04.py] | [4.txt](samples/2024/4.txt) | [Rust test][rust-2024-4] | [Rust test][rust-2024-4] |
| [2024/05/1](2024/day05.kg) | [05.py][python-2024-05.py] | [5.txt](samples/2024/5.txt) | [Rust test][rust-2024-5] | [Rust test][rust-2024-5] |
| [2024/05/2](2024/day05.kg) | [05.py][python-2024-05.py] | [5.txt](samples/2024/5.txt) | [Rust test][rust-2024-5] | [Rust test][rust-2024-5] |
| [2024/06/1](2024/day06.kg) | [06.py][python-2024-06.py] | [6.txt](samples/2024/6.txt) | [Rust test][rust-2024-6] | [Rust test][rust-2024-6] |
| [2024/06/2](2024/day06.kg) | [06.py][python-2024-06.py] | [6.txt](samples/2024/6.txt) | [Rust test][rust-2024-6] | [Rust test][rust-2024-6] |
| [2024/07/1](2024/day07.kg) | [07.py][python-2024-07.py] | [7.txt](samples/2024/7.txt) | [Rust test][rust-2024-7] | [Rust test][rust-2024-7] |
| [2024/07/2](2024/day07.kg) | [07.py][python-2024-07.py] | [7.txt](samples/2024/7.txt) | [Rust test][rust-2024-7] | [Rust test][rust-2024-7] |
| [2024/08/1](2024/day08.kg) | [08.py][python-2024-08.py] | [8.txt](samples/2024/8.txt) | [Rust test][rust-2024-8] | [Rust test][rust-2024-8] |
| [2024/08/2](2024/day08.kg) | [08.py][python-2024-08.py] | [8.txt](samples/2024/8.txt) | [Rust test][rust-2024-8] | [Rust test][rust-2024-8] |
| [2024/09/1](2024/day09.kg) | [09.py][python-2024-09.py] | [9.txt](samples/2024/9.txt) | [Rust test][rust-2024-9] | [Rust test][rust-2024-9] |
| [2024/09/2](2024/day09.kg) | [09.py][python-2024-09.py] | [9.txt](samples/2024/9.txt) | [Rust test][rust-2024-9] | [Rust test][rust-2024-9] |
| [2024/10/1](2024/day10.kg) | [10.py][python-2024-10.py] | [10.txt](samples/2024/10.txt) | [Rust test][rust-2024-10] | [Rust test][rust-2024-10] |
| [2024/10/2](2024/day10.kg) | [10.py][python-2024-10.py] | [10.txt](samples/2024/10.txt) | [Rust test][rust-2024-10] | [Rust test][rust-2024-10] |
| [2024/11/1](2024/day11.kg) | [11.py][python-2024-11.py] | [11.txt](samples/2024/11.txt) | [Rust test][rust-2024-11] | [Rust test][rust-2024-11] |
| [2024/11/2](2024/day11.kg) | [11.py][python-2024-11.py] | [11.txt](samples/2024/11.txt) | [Rust test][rust-2024-11] | [Rust test][rust-2024-11] |
| [2024/12/1](2024/day12.kg) | [12.py][python-2024-12.py] | [12.txt](samples/2024/12.txt) | [Rust test][rust-2024-12] | [Rust test][rust-2024-12] |
| [2024/12/2](2024/day12.kg) | [12.py][python-2024-12.py] | [12.txt](samples/2024/12.txt) | [Rust test][rust-2024-12] | [Rust test][rust-2024-12] |
| [2024/13/1](2024/day13.kg) | [13.py][python-2024-13.py] | [13.txt](samples/2024/13.txt) | [Rust test][rust-2024-13] | [Rust test][rust-2024-13] |
| [2024/13/2](2024/day13.kg) | [13.py][python-2024-13.py] | [13.txt](samples/2024/13.txt) | [Rust test][rust-2024-13] | [Rust test][rust-2024-13] |
| [2024/14/1](2024/day14.kg) | [14.py][python-2024-14.py] | [14.txt](samples/2024/14.txt) | [14.py][python-2024-14.py] | [14.py][python-2024-14.py] run on the fixture |
| [2024/14/2](2024/day14.kg) | [14.py][python-2024-14.py] | [14b.txt](samples/2024/14b.txt) | Project-authored tree at tick 3 | [14.py][python-2024-14.py] run on the fixture |
| [2024/15/1](2024/day15.kg) | [15.py][python-2024-15.py] | [15.txt](samples/2024/15.txt) | [Rust test][rust-2024-15] | [Rust test][rust-2024-15] |
| [2024/15/2](2024/day15.kg) | [15.py][python-2024-15.py] | [15.txt](samples/2024/15.txt) | [Rust test][rust-2024-15] | [Rust test][rust-2024-15] |
| [2024/16/1](2024/day16.kg) | [16.py][python-2024-16.py] | [16.txt](samples/2024/16.txt) | [Rust test][rust-2024-16] | [Rust test][rust-2024-16] |
| [2024/16/2](2024/day16.kg) | [16.py][python-2024-16.py] | [16.txt](samples/2024/16.txt) | [Rust test][rust-2024-16] | [Rust test][rust-2024-16] |
| [2024/17/1](2024/day17.kg) | [17.py][python-2024-17.py] | [17.txt](samples/2024/17.txt) | [Rust test][rust-2024-17] | [Rust test][rust-2024-17] |
| [2024/17/2](2024/day17.kg) | [17.py][python-2024-17.py] | [17b.txt](samples/2024/17b.txt) | [Rust test][rust-2024-17] | [Rust test][rust-2024-17] |
| [2024/18/1](2024/day18.kg) | [18.py][python-2024-18.py] | [18.txt](samples/2024/18.txt) | [18.py][python-2024-18.py] | [18.py][python-2024-18.py] run on the fixture |
| [2024/18/2](2024/day18.kg) | [18.py][python-2024-18.py] | [18.txt](samples/2024/18.txt) | [18.py][python-2024-18.py] | [18.py][python-2024-18.py] run on the fixture |
| [2024/19/1](2024/day19.kg) | [19.py][python-2024-19.py] | [19.txt](samples/2024/19.txt) | [Rust test][rust-2024-19] | [Rust test][rust-2024-19] |
| [2024/19/2](2024/day19.kg) | [19.py][python-2024-19.py] | [19.txt](samples/2024/19.txt) | [Rust test][rust-2024-19] | [Rust test][rust-2024-19] |
| [2024/20/1](2024/day20.kg) | [20.py][python-2024-20.py] | [20.txt](samples/2024/20.txt) | [20.py][python-2024-20.py] | [20.py][python-2024-20.py] run on the fixture |
| [2024/20/2](2024/day20.kg) | [20.py][python-2024-20.py] | [20.txt](samples/2024/20.txt) | [20.py][python-2024-20.py] | [20.py][python-2024-20.py] run on the fixture |
| [2024/21/1](2024/day21.kg) | [21.py][python-2024-21.py] | [21.txt](samples/2024/21.txt) | [Rust test][rust-2024-21] | [Rust test][rust-2024-21] |
| [2024/21/2](2024/day21.kg) | [21.py][python-2024-21.py] | [21.txt](samples/2024/21.txt) | [Rust test][rust-2024-21] | [Rust test][rust-2024-21] |
| [2024/22/1](2024/day22.kg) | [22.py][python-2024-22.py] | [22.txt](samples/2024/22.txt) | [Rust test][rust-2024-22] | [Rust test][rust-2024-22] |
| [2024/22/2](2024/day22.kg) | [22.py][python-2024-22.py] | [22b.txt](samples/2024/22b.txt) | [Rust test][rust-2024-22] | [Rust test][rust-2024-22] |
| [2024/23/1](2024/day23.kg) | [23.py][python-2024-23.py] | [23.txt](samples/2024/23.txt) | [Rust test][rust-2024-23] | [Rust test][rust-2024-23] |
| [2024/23/2](2024/day23.kg) | [23.py][python-2024-23.py] | [23.txt](samples/2024/23.txt) | [Rust test][rust-2024-23] | [Rust test][rust-2024-23] |
| [2024/24/1](2024/day24.kg) | [24.py][python-2024-24.py] | [24.txt](samples/2024/24.txt) | [Rust test][rust-2024-24] | [Rust test][rust-2024-24] |
| [2024/24/2](2024/day24.kg) | [24.py][python-2024-24.py] | [24b.txt](samples/2024/24b.txt) | Project-authored six-bit adder | Four known output exchanges in the [manifest](2024/cases.json) |
| [2024/25/1](2024/day25.kg) | [25.py][python-2024-25.py] | [25.txt](samples/2024/25.txt) | [Rust test][rust-2024-25] | [Rust test][rust-2024-25] |

Puzzle days · [01][puzzle-2024-1] · [02][puzzle-2024-2] · [03][puzzle-2024-3] · [04][puzzle-2024-4] · [05][puzzle-2024-5] · [06][puzzle-2024-6] · [07][puzzle-2024-7] · [08][puzzle-2024-8] · [09][puzzle-2024-9] · [10][puzzle-2024-10] · [11][puzzle-2024-11] · [12][puzzle-2024-12] · [13][puzzle-2024-13] · [14][puzzle-2024-14] · [15][puzzle-2024-15] · [16][puzzle-2024-16] · [17][puzzle-2024-17] · [18][puzzle-2024-18] · [19][puzzle-2024-19] · [20][puzzle-2024-20] · [21][puzzle-2024-21] · [22][puzzle-2024-22] · [23][puzzle-2024-23] · [24][puzzle-2024-24] · [25][puzzle-2024-25]

[python-2024-01.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/01.py
[puzzle-2024-1]: https://adventofcode.com/2024/day/1
[rust-2024-1]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day01.rs
[python-2024-02.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/02.py
[puzzle-2024-2]: https://adventofcode.com/2024/day/2
[rust-2024-2]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day02.rs
[python-2024-03.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/03.py
[puzzle-2024-3]: https://adventofcode.com/2024/day/3
[rust-2024-3]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day03.rs
[python-2024-04.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/04.py
[puzzle-2024-4]: https://adventofcode.com/2024/day/4
[rust-2024-4]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day04.rs
[python-2024-05.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/05.py
[puzzle-2024-5]: https://adventofcode.com/2024/day/5
[rust-2024-5]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day05.rs
[python-2024-06.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/06.py
[puzzle-2024-6]: https://adventofcode.com/2024/day/6
[rust-2024-6]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day06.rs
[python-2024-07.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/07.py
[puzzle-2024-7]: https://adventofcode.com/2024/day/7
[rust-2024-7]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day07.rs
[python-2024-08.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/08.py
[puzzle-2024-8]: https://adventofcode.com/2024/day/8
[rust-2024-8]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day08.rs
[python-2024-09.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/09.py
[puzzle-2024-9]: https://adventofcode.com/2024/day/9
[rust-2024-9]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day09.rs
[python-2024-10.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/10.py
[puzzle-2024-10]: https://adventofcode.com/2024/day/10
[rust-2024-10]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day10.rs
[python-2024-11.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/11.py
[puzzle-2024-11]: https://adventofcode.com/2024/day/11
[rust-2024-11]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day11.rs
[python-2024-12.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/12.py
[puzzle-2024-12]: https://adventofcode.com/2024/day/12
[rust-2024-12]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day12.rs
[python-2024-13.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/13.py
[puzzle-2024-13]: https://adventofcode.com/2024/day/13
[rust-2024-13]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day13.rs
[python-2024-14.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/14.py
[puzzle-2024-14]: https://adventofcode.com/2024/day/14
[python-2024-15.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/15.py
[puzzle-2024-15]: https://adventofcode.com/2024/day/15
[rust-2024-15]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day15.rs
[python-2024-16.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/16.py
[puzzle-2024-16]: https://adventofcode.com/2024/day/16
[rust-2024-16]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day16.rs
[python-2024-17.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/17.py
[puzzle-2024-17]: https://adventofcode.com/2024/day/17
[rust-2024-17]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day17.rs
[python-2024-18.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/18.py
[puzzle-2024-18]: https://adventofcode.com/2024/day/18
[python-2024-19.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/19.py
[puzzle-2024-19]: https://adventofcode.com/2024/day/19
[rust-2024-19]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day19.rs
[python-2024-20.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/20.py
[puzzle-2024-20]: https://adventofcode.com/2024/day/20
[python-2024-21.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/21.py
[puzzle-2024-21]: https://adventofcode.com/2024/day/21
[rust-2024-21]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day21.rs
[python-2024-22.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/22.py
[puzzle-2024-22]: https://adventofcode.com/2024/day/22
[rust-2024-22]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day22.rs
[python-2024-23.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/23.py
[puzzle-2024-23]: https://adventofcode.com/2024/day/23
[rust-2024-23]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day23.rs
[python-2024-24.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/24.py
[puzzle-2024-24]: https://adventofcode.com/2024/day/24
[rust-2024-24]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day24.rs
[python-2024-25.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2024/25.py
[puzzle-2024-25]: https://adventofcode.com/2024/day/25
[rust-2024-25]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2024/day25.rs

## 2025

[Expected answers and parameters](2025/cases.json)

| Klong part | Upstream solution | Local input | Input origin | Expected answer source |
|---|---|---|---|---|
| [2025/01/1](2025/day01.kg) | [01.py][python-2025-01.py] | [1.txt](samples/2025/1.txt) | [Rust test][rust-2025-1] | [Rust test][rust-2025-1] |
| [2025/01/2](2025/day01.kg) | [01.py][python-2025-01.py] | [1.txt](samples/2025/1.txt) | [Rust test][rust-2025-1] | [Rust test][rust-2025-1] |
| [2025/02/1](2025/day02.kg) | [02.py][python-2025-02.py] | [2.txt](samples/2025/2.txt) | [Rust test][rust-2025-2] | [Rust test][rust-2025-2] |
| [2025/02/2](2025/day02.kg) | [02.py][python-2025-02.py] | [2.txt](samples/2025/2.txt) | [Rust test][rust-2025-2] | [Rust test][rust-2025-2] |
| [2025/03/1](2025/day03.kg) | [03.py][python-2025-03.py] | [3.txt](samples/2025/3.txt) | [Rust test][rust-2025-3] | [Rust test][rust-2025-3] |
| [2025/03/2](2025/day03.kg) | [03.py][python-2025-03.py] | [3.txt](samples/2025/3.txt) | [Rust test][rust-2025-3] | [Rust test][rust-2025-3] |
| [2025/04/1](2025/day04.kg) | [04.py][python-2025-04.py] | [4.txt](samples/2025/4.txt) | [Rust test][rust-2025-4] | [Rust test][rust-2025-4] |
| [2025/04/2](2025/day04.kg) | [04.py][python-2025-04.py] | [4.txt](samples/2025/4.txt) | [Rust test][rust-2025-4] | [Rust test][rust-2025-4] |
| [2025/05/1](2025/day05.kg) | [05.py][python-2025-05.py] | [5.txt](samples/2025/5.txt) | [Rust test][rust-2025-5] | [Rust test][rust-2025-5] |
| [2025/05/2](2025/day05.kg) | [05.py][python-2025-05.py] | [5.txt](samples/2025/5.txt) | [Rust test][rust-2025-5] | [Rust test][rust-2025-5] |
| [2025/06/1](2025/day06.kg) | [06.py][python-2025-06.py] | [6.txt](samples/2025/6.txt) | [Rust test][rust-2025-6] | [Rust test][rust-2025-6] |
| [2025/06/2](2025/day06.kg) | [06.py][python-2025-06.py] | [6.txt](samples/2025/6.txt) | [Rust test][rust-2025-6] | [Rust test][rust-2025-6] |
| [2025/07/1](2025/day07.kg) | [07.py][python-2025-07.py] | [7.txt](samples/2025/7.txt) | [Rust test][rust-2025-7] | [Rust test][rust-2025-7] |
| [2025/07/2](2025/day07.kg) | [07.py][python-2025-07.py] | [7.txt](samples/2025/7.txt) | [Rust test][rust-2025-7] | [Rust test][rust-2025-7] |
| [2025/08/1](2025/day08.kg) | [08.py][python-2025-08.py] | [8.txt](samples/2025/8.txt) | [Rust test][rust-2025-8] | [Rust test][rust-2025-8] |
| [2025/08/2](2025/day08.kg) | [08.py][python-2025-08.py] | [8.txt](samples/2025/8.txt) | [Rust test][rust-2025-8] | [Rust test][rust-2025-8] |
| [2025/09/1](2025/day09.kg) | [09.py][python-2025-09.py] | [9.txt](samples/2025/9.txt) | [Rust test][rust-2025-9] | [Rust test][rust-2025-9] |
| [2025/09/2](2025/day09.kg) | [09.py][python-2025-09.py] | [9.txt](samples/2025/9.txt) | [Rust test][rust-2025-9] | [Rust test][rust-2025-9] |
| [2025/10/1](2025/day10.kg) | [10.py][python-2025-10.py] | [10.txt](samples/2025/10.txt) | [Rust test][rust-2025-10] | [Rust test][rust-2025-10] |
| [2025/10/2](2025/day10.kg) | [10.py][python-2025-10.py] | [10.txt](samples/2025/10.txt) | [Rust test][rust-2025-10] | [Rust test][rust-2025-10] |
| [2025/11/1](2025/day11.kg) | [11.py][python-2025-11.py] | [11.txt](samples/2025/11.txt) | [Rust test][rust-2025-11] | [Rust test][rust-2025-11] |
| [2025/11/2](2025/day11.kg) | [11.py][python-2025-11.py] | [11b.txt](samples/2025/11b.txt) | [Rust test][rust-2025-11] | [Rust test][rust-2025-11] |
| [2025/12/1](2025/day12.kg) | [12.py][python-2025-12.py] | [12.txt](samples/2025/12.txt) | [puzzle example][puzzle-2025-12] | [puzzle example][puzzle-2025-12] |

Puzzle days · [01][puzzle-2025-1] · [02][puzzle-2025-2] · [03][puzzle-2025-3] · [04][puzzle-2025-4] · [05][puzzle-2025-5] · [06][puzzle-2025-6] · [07][puzzle-2025-7] · [08][puzzle-2025-8] · [09][puzzle-2025-9] · [10][puzzle-2025-10] · [11][puzzle-2025-11] · [12][puzzle-2025-12]

[python-2025-01.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/01.py
[puzzle-2025-1]: https://adventofcode.com/2025/day/1
[rust-2025-1]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day01.rs
[python-2025-02.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/02.py
[puzzle-2025-2]: https://adventofcode.com/2025/day/2
[rust-2025-2]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day02.rs
[python-2025-03.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/03.py
[puzzle-2025-3]: https://adventofcode.com/2025/day/3
[rust-2025-3]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day03.rs
[python-2025-04.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/04.py
[puzzle-2025-4]: https://adventofcode.com/2025/day/4
[rust-2025-4]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day04.rs
[python-2025-05.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/05.py
[puzzle-2025-5]: https://adventofcode.com/2025/day/5
[rust-2025-5]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day05.rs
[python-2025-06.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/06.py
[puzzle-2025-6]: https://adventofcode.com/2025/day/6
[rust-2025-6]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day06.rs
[python-2025-07.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/07.py
[puzzle-2025-7]: https://adventofcode.com/2025/day/7
[rust-2025-7]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day07.rs
[python-2025-08.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/08.py
[puzzle-2025-8]: https://adventofcode.com/2025/day/8
[rust-2025-8]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day08.rs
[python-2025-09.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/09.py
[puzzle-2025-9]: https://adventofcode.com/2025/day/9
[rust-2025-9]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day09.rs
[python-2025-10.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/10.py
[puzzle-2025-10]: https://adventofcode.com/2025/day/10
[rust-2025-10]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day10.rs
[python-2025-11.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/11.py
[puzzle-2025-11]: https://adventofcode.com/2025/day/11
[rust-2025-11]: https://github.com/maneatingape/advent-of-code-rust/blob/a3a2a860d28b574eb05989af86ee7f23db495ba0/tests/year2025/day11.rs
[python-2025-12.py]: https://github.com/maread99/aoc/blob/3937e4a7c108248fd7637ae36ef924ff72717226/2025/12.py
[puzzle-2025-12]: https://adventofcode.com/2025/day/12

## Adaptation notes

- The references also identify earlier contributors. Marcus Read credits hyper-neutrino for the split-valve search and discusses their spring solver. He credits mebeim for the robot-cap pruning rule. The pinned source files preserve those explanations and their links. [Valve search][python-2022-day16_rev2.py] · [Spring counts][python-2023-12.py] · [Factory search][python-2022-day19.py]
- The original 2022 day 2 part 2 has two additional compatibility variants. Its three versions are covered by one scorer. The original 2023 programs also share a helper file. [Numeric variant](https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/2b_compat.kg) · [String variant](https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/22/2b_compat_strings.kg) · [2023 helper](https://github.com/briangu/aoc/blob/defeb2e63ade7aed486936e5de18dfb7f2db7907/23/common.kg)
- The original 2022 day 15 part 1 stopped at a scan. The port merges row intervals and handles the search bound separately in part 2. For 2022 day 19 the fixture joins each blueprint onto one line. For day 22 it restores the space represented by a leading `@` in the Rust test.
- The 2022 day 21 port solves for the unknown arithmetically. Day 22 folds the input's cube net. The 2023 day 25 port finds the cut from the graph. These replace decisions the references made for particular inputs.
- The 2024 day 14 tree test uses a generated alignment, and day 24 uses four known wire swaps. The 2025 day 12 port adds exact packing for regions that survive the initial area and bounding-box checks. Its official example exercises that path.
