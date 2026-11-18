# Klong functions as FPGA hardware

A function of the [static profile](static-profile.md) becomes an FPGA kernel, and you call it from a KlongPy session with ordinary arrays. This page is for someone who has an FPGA card, or wants to know what the flow produces and how fast it is.

!!! warning "Experimental"
    This has been run on one card, an Alveo U250. The commands, the names `.fpga` and `kgpy fpga build`, and the file formats may change. The first kernel measured (`fir`) is slower than numpy. The strategy kernel is about ten times faster than a 32-thread CPU running the same program as C. Both are measured [below](#results), with the cases where the CPU wins.

The same source runs three ways: interpreted, as C (see [Compiling Klong to C](embedded.md)) and on the card, with the same answers. Two kinds of kernel exist. A *streaming kernel*, the default, takes one record per clock cycle when the program allows it. A *wrapped kernel* (`--wrapped`) runs the program's C unchanged once per record and is slower.

The tools need no Python packages beyond KlongPy's own. The machine with the card also needs the Xilinx runtime (`pyxrt`, which is not on PyPI), and the build machine needs Vitis.

## Tested setup

| | |
|---|---|
| Card | Alveo U250 |
| Platform | `xilinx_u250_gen3x16_xdma_4_1_202210_1` |
| Toolchain | Vitis and Vitis HLS 2022.2 |
| Host | Ryzen 9 7950X3D (16 cores, 32 threads), gcc 9.4, numpy 1.24 |
| Link | the card sits on a PCIe link of one lane, so transfers are slower than on a full slot |
| Dates | `fir` image built 2026-11-16, strategy image measured 2026-11-17 |
| Images run | two: `fir` (a wrapped kernel) and the strategy (a streaming kernel) |

## What qualifies

A program is a candidate when it is inside the static profile (`kgpy check FILE`) and an `.entry` of it is a function that

* takes at least one argument and returns a value, or returns `void` and is the driver of a [summary pair](static-profile.md#a-summary-entry);
* takes `int`, `float` and `fixed<W,I>` arguments and arrays of them with fixed shapes, not `str`;
* does not reach an `.extern`: the card has nothing to call;
* does not mix `int` and `float` arguments in one entry, for the host side (`.fpga` refuses such an entry; the build accepts it);
* computes in 32-bit `float` in a float program (the card has no `--float64`); `int` is 32 bits unless the program was built with `--int64`.

Every `.entry` becomes one kernel. The program's globals are the kernel's state between calls. Two entries of one program are two kernels with a state each, and they do not share globals. The exception is a summary pair, which is one kernel.

## The flow

### 1. Emit

`kgpy fpga source FILE... -o DIR` writes the HLS C++ of every entry, a C test bench, a script for Vitis HLS and `manifest.json`. It takes seconds and needs none of the tools.

<!-- check-docs: link examples -->
```console
$ kgpy fpga source examples/embedded/fir.kg -o out/
examples/embedded/fir.kg: wrote out/fir.c, out/fir.h, out/fir_run_tb.cpp, out/fir_run_top.cpp, out/hls_run.tcl, out/kg_hls_stub.h, out/klong_rt.h, out/manifest.json
  kernel fir_run_kernel: entry run (float[256] -> float[241]); record in 1024 bytes, out 964 bytes
```

With `KLONG_FPGA_BUILD=user@host` set, `--synth` runs C synthesis on that machine and prints the estimates.

### 2. Build, with the timing gate

`kgpy fpga build FILE... [--clock-mhz N]` builds the bitstream on the machine named by `KLONG_FPGA_BUILD` and keeps it in a cache there. The cache key is a hash of the program, the emitter version, the clock, the platform and the tool version. Several files make one image, so build together what you will use together:

```
kgpy fpga build examples/embedded/mavg.kg examples/embedded/fir.kg
kgpy fpga status                # shows the build
kgpy fpga build --wait          # waits for it
```

A source that has been built is not built again. A source that failed timing is not retried; the message gives the slack and a clock to try. A build is a full Vitis and Vivado run. One image of three small programs (four kernels) took 1 hour 27 minutes. Run one build at a time: `kgpy fpga build` refuses to start while a Vivado, `v++` or Vitis HLS process of any user is running on the machine.

A bitstream is loaded only if its routed timing was met: no failing endpoints and a worst negative slack at or above zero. The build records the result in `build.json` next to the image. An image that fails is renamed `*.timing-failed.xclbin`. The loader refuses that name, any image without a passing `build.json`, and any image whose SHA-256 differs from the recorded one.

### 3. Bind and call from Klong

`.fpga("fir.kg")` finds the image of that source, loads it on the card and returns a dictionary from entry name to function. A source with no image is an error that names the exact build command. `.fpga` never runs an image that was built from other source.

With `KLONG_FPGA=sim`, `.fpga` needs no card and no build. It compiles the generated HLS code with the system C and C++ compilers (the HLS pragmas are ignored) and calls it with the same records and status word as the card. Its results show that the program and the binding work. They say nothing about speed.

`fir.kg` is a 16-tap filter over a block of 256 samples that gives 241 outputs. The program below loads it interpreted (`.l` defines `run`), binds the kernel (`fir.run`), runs three blocks both ways, prints the shape of the result, and prints 1 if the largest squared difference is below 1e-9.

<!-- check-docs: link examples; requires-cmd cc c++ -->
```console
$ cat > fir_hw.kg <<'EOF'
> .l("examples/embedded/fir.kg")
> fir::.fpga("examples/embedded/fir.kg")
> x::{256#x}'1.0+!3
> d::run'x
> d::d-fir.run(x)
> .p(^d)
> .p((|/,/d*d)<1e-9)
> EOF
$ KLONG_FPGA=sim kgpy fir_hw.kg
[3 241]
1
```

`run'x` is the interpreted program over three blocks; `fir.run(x)` is one run of three calls on the kernel. The array `x` has one more dimension than the entry's argument (`float[256]`), which means three calls in one run, and the result has the same extra dimension. A value of exactly `float[256]` is one call and gives one result of `float[241]`.

State lives in the kernel. `mavg.kg` keeps the last four samples:

<!-- check-docs: continue; requires-cmd cc c++ -->
```console
$ cat > mavg_hw.kg <<'EOF'
> mavg::.fpga("examples/embedded/mavg.kg")
> .p(mavg.step([4.0 8.0 12.0 16.0 20.0]))
> .p(mavg.step(4.0))
> .fpga.reset(mavg.step)
> .p(mavg.step(4.0))
> .p(.fpga.info()@"mode")
> EOF
$ KLONG_FPGA=sim kgpy mavg_hw.kg
[1.0 3.0 6.0 10.0 14.0]
13.0
1.0
sim
```

The first call after `.fpga` starts the program from its initial state; the five ticks are one run. The next call goes on from the window the run left. `.fpga.reset(k)` makes the next call of `k` start from the initial state again.

### 4. Lanes and parameters

`--lanes N` builds N copies of the program, called lanes, that read one input stream. Each parameter (declared with `.param`) is then an array of N values, set with `.fpga.set` between runs. `.fpga.lanes("crossover.kg";n)` binds the image built with `--lanes n`.

On the simulator, with two lanes (`fast` 2 and 1, `slow` 4 and 3) and the 20 ticks of a test series whose answer was worked out on paper:

<!-- check-docs: link examples; requires-cmd cc c++ -->
```console
$ cat > strategy.kg <<'EOF'
> s::.fpga.lanes("examples/hardware/crossover.kg";2)
> .fpga.set(s;"fast";[2 1])
> .fpga.set(s;"slow";[4 3])
> .fpga.set(s;"stop";[1.5 1.5])
> t::[10 10 10 10 11 9.25 9.5 9.75 11.5 12 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17]
> .p(crossover.step(t))
> .p(crossover.summary())
> EOF
$ KLONG_FPGA=sim kgpy strategy.kg
[[1.75 3.75 3.0] [5.0 2.25 3.0]]
[[1.75 3.75 3.0] [5.0 2.25 3.0]]
```

Each row is one lane: profit, largest drawdown, number of trades. The first lane is that series
(`tests/static/test_crossover.py`); the same account comes out of the Python evaluator of the static profile (`klongpy.static.eval_ir`), the emitted C, the kernel in
C simulation and this simulator, bit for bit.

### 5. The summary pattern

A result per tick would cost more than the computation. With 32 lanes, a result of three 64-bit words per tick is 96 words, and a kernel writes one word per clock cycle. So a strategy is written as a *summary pair*: `step` returns `void` and updates the state, and `summary` reads the state at the end. The kernel runs n calls of `step`, writes nothing for them, then writes the one record of `summary`, per lane. `examples/hardware/crossover.kg` is a moving-average crossover strategy written this way; the rules for the pair are in [the static profile](static-profile.md#a-summary-entry).

```
.param("fast";"int";5)                            :" the fast window, 1 to 32 ticks "
.param("slow";"int";20)
.param("stop";"fixed<32,16>";1)                   :" close a position this far under water "

step::{...update the account...;0}                .entry("step";"fixed<32,16> -> void")
summary::{profit,maxdd,wide(trades)}              .entry("summary";"-> fixed<64,32>[3]")
```

This is a fragment: `step`, `profit`, `maxdd`, `trades` and `wide` are defined in the file. Three design choices let it fit 32 lanes at 250 MHz:

* **No multiplication in what a tick carries to the next.** A multiplication of two `fixed<32,16>` numbers in the feedback of a state variable limited a one-lane moving average to 228 MHz in C synthesis. In the strategy the sums change by an add and a subtract, the position by comparisons and the profit by an add.
* **One process per lane.** A single pipelined loop that holds every lane stalls on one signal, an empty input FIFO, which drives the clock enable of every register of every lane. That net cannot be replicated or pipelined and routes at several times the clock period. With one dataflow process per lane the net reaches a few thousand registers. The input words are copied to every lane's stream and the lanes' records are collected in lane order at the end.
* **Parameters as a table.** The host runtime can write a scalar to a kernel's register but not an array. The server writes the parameters of all lanes into a table in card memory that the kernel reads when it starts.

Building and replaying it on the card:

```
kgpy fpga build examples/hardware/crossover.kg --lanes 32 --clock-mhz 250
python examples/hardware/crossover_bench.py gen history.bin params.bin --ticks 10000000 --lanes 32
KLONG_FPGA_HOST=user@host python examples/hardware/crossover_card.py history.bin params.bin
```

## Settings

| Setting | Meaning | Default |
|---|---|---|
| `KLONG_FPGA=sim` | use the simulator on this machine; no card, no image | unset |
| `KLONG_FPGA_HOST=user@host` | the machine with the card, reached by key-based ssh; the server `klongpy/hw/server.py` is started there with Python 3.8, numpy and pyxrt | unset; with `pyxrt` importable, the card on this machine |
| `KLONG_FPGA_BUILD=user@host` | where `kgpy fpga source --synth` and `kgpy fpga build` run | unset |
| `KLONG_FPGA_ROOT` | the directory on the build and card machine under which everything is written (`work/`, `builds/`, `server/`, `venv/`, `build.lock`); absolute, or starting with `~/`, the home directory of that machine; a path outside it is refused | `~/klong-hw` |
| `KLONG_FPGA_VITIS_SETTINGS` | the `settings64.sh` of Vitis that a build sources before `v++` and `vitis_hls` | none: `v++` and `vitis_hls` must be on the `PATH` of the machine, or the build stops and names this variable |
| `KLONG_FPGA_HLS_SETTINGS` | the `settings64.sh` of Vitis HLS that `--synth` and the remote tests source before `vitis_hls` | none: `vitis_hls` must be on the `PATH` |
| `KLONG_FPGA_PLATFORM_REPO_PATHS` | the directory that holds the U250 platform, exported to the build as `PLATFORM_REPO_PATHS` | none: a build stops with an error naming this variable unless `PLATFORM_REPO_PATHS` is already set on the machine |

The build machine and the card machine may be the same machine but are separate settings. Nothing of this is imported when KlongPy starts. The settings are read where `kgpy` runs and used on the machine named by `KLONG_FPGA_BUILD` or `KLONG_FPGA_HOST`. The hash that names a build directory does not include the root, so images built under one root are found again when the root is set to that directory.

### Trust

The server runs on the card machine with your ssh identity, and the build runs there as you. Loading an image replaces what was on the card, including an image of someone else's, after checking that no other process has the card open. The loader does not program the base shell and never loads an image whose timing failed. All of its files are under `KLONG_FPGA_ROOT` (default `~/klong-hw/`) on that machine. Nothing is installed outside it, and nothing needs sudo.

## Calling a kernel

`h::.fpga("fir.kg")` binds each entry twice: as `h@"run"` in the returned dictionary (`g::h@"run"; g(x)`), and as the name `fir.run`, the file name without `.kg`, then a dot, then the entry. A list of paths, `.fpga(["mavg.kg" "fir.kg"])`, binds the programs of an image built from several files.

* A call checks the number, type and shape of its arguments against the entry's signature before anything is sent, with the errors `call_entry` gives: a wrong type is a `type` error, a wrong shape a `shape` error, a wrong count an `arity` error, an `int` outside the kernel's 32 bits a `range` error.
* An argument with one extra leading dimension `n` is `n` calls in one run, and the result has the same leading dimension. All arguments of a call must agree on `n`.
* A float result is returned as float64, an int as int64, in the array type of the session's backend (a torch tensor on the torch backend). The arrays cross to the card as numpy.
* `.fpga.put(k;x)` uploads the arguments of calls of `k` once and returns a buffer. For a kernel with several arguments `x` is a list of them. `k(b)` runs the stored calls without sending them again, `.fpga.get(b)` downloads them and `.fpga.free(b)` frees them. Buffers are freed when another image is loaded and when the session ends.
* `.fpga.lanes("fir.kg";n)` is `.fpga` for the image built with `--lanes n`. Every parameter of the program (declared with `.param`) is then an array of n values, one per lane, set with `.fpga.set(h;"name";values)` between runs, and a result has a leading dimension n. `.fpga.set` converts the value as `klongpy.static.set_param` does for the interpreter, and an unknown name, a wrong shape (one value per lane) or a value that does not fit the register is an error. See [the summary pattern](#5-the-summary-pattern).
* `.fpga.info()` is a dictionary: `mode` (`card`, `local` or `sim`), `device`, `shell`, the loaded `image` and its `key`, the `kernels`, the `build` record (routed timing, utilisation, build time), and the number and size of the `buffers`.

Loading an image replaces whatever the card held, and one session holds one image: a second `.fpga` that needs another image loads it, and the kernels bound earlier then raise an error that says to call `.fpga` again. Bind everything that is used together in one image.

### Errors

All are `system` errors (a subclass of `KlongSystemError`) except the first, which is the checker's own, and the last two rows, which are `name` and `range` errors.

| Situation | What you see |
|---|---|
| The program is outside the profile, or an entry cannot be a kernel | a `static` error with the position and the rule |
| No hardware: none of `KLONG_FPGA=sim`, `KLONG_FPGA_HOST`, `pyxrt` | a `system` error naming the three ways to get one |
| No image for this source, or the build failed timing, or it is still being built | a `system` error with the exact `kgpy fpga build ...` command |
| The server refused an image (timing, no passing `build.json`, hash mismatch) | a `system` error; nothing is loaded |
| The card is in use by another process, or a unit is not idle | a `system` error; nothing is loaded |
| The connection dropped or the server stopped answering | a `system` error; call `.fpga` again to reconnect |
| `.fpga.set` of a name the program does not declare with `.param` | a `name` error that lists the parameters |
| A trap during a run: an index out of range, an integer division by zero | a `range` error naming the entry; the results of that run are not returned, and the next run starts clean |

The trap is the compiled program's trap: the kernel does not stop, it sets a flag, and the host raises after the run. The interpreter raises at the first call that does it.

<!-- check-docs: link examples -->
```console
$ cat > no_hw.kg <<'EOF'
> .fpga("examples/embedded/fir.kg")
> EOF
$ env -u KLONG_FPGA -u KLONG_FPGA_HOST kgpy no_hw.kg
no_hw.kg:1:1: system error: no hardware to run on: set KLONG_FPGA_HOST=user@host to use the Alveo card on that machine over ssh, run KlongPy on the machine with the card (pyxrt importable), or set KLONG_FPGA=sim for the simulator (no hardware)
  .fpga("examples/embedded/fir.kg")
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

## Results

### The `fir` image

A 16-tap filter over blocks of 256 samples, built 2026-11-16 at 300 MHz (met, worst slack 0.027 ns), a wrapped kernel: the hardware runs the program's C unchanged. Best of 3 to 5 runs (2 for 100000 blocks). `put`, `kernel` and `get` are the server's clock on the card machine; `card wall` is what the client saw, including ssh.

| blocks | put (s) | kernel (s) | get (s) | card wall (s) | numpy, client machine (s) | numpy, card machine CPU (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000045 | 0.000084 | 0.000015 | 0.002701 | 0.000011 | 0.000015 |
| 1000 | 0.001590 | 0.013105 | 0.001688 | 0.040323 | 0.004779 | 0.005158 |
| 100000 | 0.144330 | 1.294230 | 0.158533 | 3.400952 | 1.025725 | 1.046810 |

The kernel takes about 12.9 microseconds per block, about 3886 cycles at 300 MHz. A wrapped kernel cannot do a floating-point sum at one result per cycle. Numpy takes about 10 microseconds per block on the same gather and weighted sum. The card is slower than numpy at every size, even before the transfer is counted. A kernel rebuilt as a shift register with a pipelined reduction is where speed would have to come from; the streaming kernel does this, and `fir` has not been rebuilt that way.

The image of `mavg.kg`, `fir.kg` and `tests/static/hw/trap.kg` (a program that traps on purpose) uses 126578 LUT (7.33 percent of the device with the shell), 199444 FF, 276.5 BRAM tiles and 18 DSP.

### The strategy kernel

Measured 2026-11-17. The image is `crossover.kg` built with `--lanes 32 --clock-mhz 250`. It passed routed timing at 250 MHz (worst setup and hold slack 0.009 ns, no failing endpoints; the margin is thin) after a build of 1 hour 54 minutes. It uses 13.6 percent of the device's LUTs (234315, the shell's 118282 included), 7.0 percent of its registers, 258 block RAM tiles and 135 DSPs. The history is 10 million synthetic ticks, a random walk with rare jumps folded into the band 5 to 15 (`crossover_bench.py gen`), held as a file on the card machine. The 32 parameter sets are 4 values of `fast`, 4 of `slow` and 2 stops. Every lane replays all 10 million ticks, 320 million lane-ticks in all.

| | time for 10 M ticks x 32 lanes | lane-ticks per second, total | ticks per second per lane (card) or per process (CPU) |
|---|---:|---:|---:|
| card, kernel only (the server's clock; best of 7) | 40.2 ms | 7.97 G | 249 M |
| card, history already on the card, wall time of the call over ssh (best of 7) | 42.7 ms | 7.50 G | 234 M |
| card, the same with the 40 MB history uploaded first | 0.56 s | 0.57 G | 18 M |
| emitted C, `gcc -O3`, one core, 32 lanes in turn (best and worst of 5) | 6.0 s to 6.8 s | 47 M to 53 M | 47 M to 53 M |
| the same, 8 processes of 4 lanes | 0.87 s | 366 M | 46 M |
| the same, 16 processes of 2 lanes (one per core; best of 5) | 0.46 s to 0.47 s | 691 M | 43 M |
| the same, 32 processes of 1 lane (one per thread; best of 5) | 0.42 s | 758 M | 24 M |
| numpy (the sums and the signal for all lanes at once, the position stepped one tick at a time), on 1 M ticks | 20.3 s for 1 M ticks | 1.6 M | 50 k |

The card replays one tick per clock cycle in every lane, 250 M a second at 250 MHz, with 32 lanes at once. Against the C that the same Klong source compiles to, it is 10 times faster than all 32 threads of the CPU (40.2 ms against 0.42 s) and 150 times faster than one core. A lane on the card is five times a core.

The upload of the history is a separate cost and the largest one when the data comes over ssh: 0.52 s for 40 MB, against 0.08 s for the transfer from memory to the card on the server's clock. The CPU's numbers include reading the history from the page cache. Numpy is not a fair opponent on long histories, because the position of a tick depends on the one before: only the sums and the signal vectorise.

The figures come from two runs. The first table is the benchmark script's default of 7 card runs (`crossover_card.py --runs`) and 5 CPU runs. The sweep below used 9 card runs. They agree within 3 percent at 10 million ticks: 40.2 ms against 40.2 ms for the kernel, 42.7 ms against 43 ms for the call, and 0.46 s against 0.47 s for 16 processes.

### Where the CPU wins

The card has a fixed cost of about 1 ms per call: its kernel takes 0.05 ms for 1000 ticks and a call from the client 1.0 ms. A computation of fewer than about 50 thousand ticks of one parameter set is faster on one core, which runs 50 M ticks a second per lane. The card needs all 32 lanes to be worth it. With 32 parameter sets it already wins at 1000 ticks (2.4 ms on one core, 1.0 ms on the card) and by 12 times at 100000 ticks (1.7 ms against 21 ms for 16 processes).

The number of lanes and every `.entry` are fixed at build time, and a build takes hours. A CPU compiles the program in a second and runs any number of lanes.

| ticks | card kernel | card wall | CPU, 1 core | CPU, 16 processes | CPU, 32 processes |
|---:|---:|---:|---:|---:|---:|
| 1 000 | 0.05 ms | 1.0 ms | 2.4 ms | 15.7 ms | 32 ms |
| 10 000 | 0.09 ms | 1.1 ms | 6.9 ms | 16.3 ms | 31 ms |
| 100 000 | 0.45 ms | 1.7 ms | 60 ms | 20.8 ms | 32 ms |
| 1 000 000 | 4.1 ms | 5.9 ms | 0.59 s | 63 ms | 67 ms |
| 10 000 000 | 40.2 ms | 43 ms | 5.9 s | 0.47 s | 0.43 s |

Card: best of 9 calls; CPU: best of 5; 32 lanes. The CPU's processes start together and each reads its ticks, so the small cases show the cost of starting 16 or 32 processes.

### Correctness

The summary of every lane after the first 100000 ticks equals the Python evaluator's (`klongpy.static.eval_ir`, which `call_entry` uses). After all 10 million ticks it equals the emitted C's, bit for bit, on the card. The hand-checked 20-tick series give the worked results in every lane. `tests/static/test_crossover_card.py` runs the hand series and 200000 ticks against the card. For `fir`, floats agree within 1e-5 relative.

## Known limits

* Only the strategy image has been run on the card with the current emitter. The image of `mavg.kg`, `fir.kg` and `trap.kg` must be built again (`kgpy fpga build examples/embedded/mavg.kg examples/embedded/fir.kg tests/static/hw/trap.kg`) before `tests/static/test_hw_card.py` and `test_hw_binding_card.py` stop skipping.
* More than 32 lanes: a build with 64 lanes in one pipelined loop failed timing (estimated worst slack -6.4 ns at 250 MHz after placement, with five nets of more than 10000 loads) and was stopped during routing. 64 lanes as one process per lane (about 290000 LUTs) has not been built.
* Hardware emulation: loading the emulated image did not complete in the time allowed. There is no co-simulation.
* There are no network ports.
* An entry that mixes `int` and `float` arguments is built but cannot be called by `.fpga`, and a summary kernel with lanes needs records of one width.
* One session holds one image. Loading an image replaces what the card held.
* Builds take hours and one image at a time.
