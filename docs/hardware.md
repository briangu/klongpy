# Klong functions as FPGA hardware

!!! warning "Experimental"
    This is new and has been run on one machine and one card, an Alveo U250. The commands, the names `.hw` and `--hw-build`, and the file formats may change. The hardware does not beat numpy on the first kernel that was measured (`fir`, see [Measured](#measured)), and it does beat a 32-thread CPU running the same program as C by about ten times on the [strategy kernel](#strategies-many-parameter-sets-over-one-tick-history), where 32 parameter sets run over one tick history; both are measured below, with where the CPU wins. What it gives you is the path itself: one Klong source that runs interpreted, as C, and on the card, with the same answers.

A function of the [static profile](static-profile.md) can be built into a hardware kernel and called from a KlongPy session with ordinary arrays. The same source runs three ways: interpreted, as C (see [Compiling Klong to C](embedded.md)), and on the card.

```
kgpy --emit-hls fir.kg -o out/       # seconds: HLS C++ and build scripts
kgpy --hw-build fir.kg               # hours: a bitstream on the build machine, cached by source
fir::.hw("fir.kg")                   # in KlongPy: bind the built kernels
y::fir.run(x)                        # runs on the card
```

## What qualifies

A program is a candidate when it is inside the static profile (`kgpy --check-static FILE`) and an `.entry` of it is a function that

* takes at least one argument and returns a value (a record in, a record out), or returns `void` and is the driver of a [summary pair](#strategies-many-parameter-sets-over-one-tick-history),
* takes `int`, `float` and `fixed<W,I>` arguments and arrays of them with fixed shapes, not `str`,
* does not reach an `.extern`: the card has nothing to call,
* does not mix `int` and `float` arguments in one entry, for the host side (`.hw` refuses such an entry; the build accepts it),
* in a float program computes in 32-bit `float` (the card has no `--float64`); `int` is 32 bits unless the program was built with `--int64`.

Every `.entry` becomes one kernel. The state of the program (its globals) is kept inside the kernel between calls. Two entries of one program are two kernels with a state each; they do not share globals, with one exception: a *summary pair*, a `void` entry with arguments and one entry without arguments that returns a value, is one kernel (see [Strategies](#strategies-many-parameter-sets-over-one-tick-history)).

## The three commands

`kgpy --emit-hls FILE... -o DIR` writes the HLS C++ of every entry, a C test bench, a script for Vitis HLS and `manifest.json`. It takes seconds and needs no machine with the tools. With `KLONG_HW_BUILD=user@host` set, `--synth` runs C synthesis there and prints the estimates.

`kgpy --hw-build FILE... [--clock-mhz N]` builds the bitstream on the machine named by `KLONG_HW_BUILD` and leaves it in a cache there, keyed by a hash of the program (its typed intermediate form), the emitter version, the clock, the platform and the tool version. Several files make one image, so everything that is going to be used together is built once: `kgpy --hw-build mavg.kg fir.kg`. `--hw-build --status` shows the build, `--hw-build --wait` waits for it. A source that has been built is not built again. A source that failed timing is not retried; the message gives the slack and a clock to try.

`.hw("fir.kg")` in KlongPy finds the image of that source, loads it on the card and returns a dictionary from entry name to function. A source with no image is an error that names the exact build command. `.hw` never runs an image that was built from other source.

### Build time

A build is a full Vitis and Vivado run: synthesis, placement and routing. The one image of three small programs (four kernels) on the U250 took 1 hour 27 minutes. Run one build at a time: `--hw-build` refuses to start while one of its own builds, or any Vivado, `v++` or Vitis HLS process of any user, is running on the machine.

### The timing gate

A bitstream is loaded only if its routed timing was met (no failing endpoints, worst negative slack at or above zero). The build script reads the routed timing report and records the result in `build.json` next to the image. An image that fails is renamed `*.timing-failed.xclbin`, and the loader refuses that name and any image without a passing `build.json` or whose SHA-256 differs from the one recorded. `.hw` raises an error for such an image; it does not load it.

## A worked example on the simulator

With `KLONG_HW=sim`, `.hw` needs no card and no build. It compiles the same generated HLS code that the hardware flow synthesises, on this machine with the system C and C++ compilers (the HLS pragmas are ignored), and calls it with the same records and the same status word as the card. The requests go through the same server code with a simulated device. It is a simulator: its results show that the program and the binding work, and say nothing about speed.

`fir.kg` is a 16-tap filter over a block of 256 samples that gives 241 outputs. The program below loads it interpreted (`.l` defines `run`), binds the kernel (`fir.run`), runs three blocks both ways and prints the largest squared difference test.

<!-- check-docs: link examples; requires-cmd cc c++ -->
```console
$ cat > fir_hw.kg <<'EOF'
> .l("examples/embedded/fir.kg")
> fir::.hw("examples/embedded/fir.kg")
> x::{256#x}'1.0+!3
> d::run'x
> d::d-fir.run(x)
> .p(^d)
> .p((|/,/d*d)<1e-9)
> EOF
$ KLONG_HW=sim kgpy fir_hw.kg
[3 241]
1
```

`run'x` is the interpreted program over three blocks; `fir.run(x)` is one run of three calls on the kernel. The array `x` has one more dimension than the entry's argument (`float[256]`), which means three calls in one run, and the result has the same extra dimension. A value of exactly `float[256]` is one call and gives one result of `float[241]`.

State lives in the kernel. `mavg.kg` keeps the last four samples:

<!-- check-docs: continue; requires-cmd cc c++ -->
```console
$ cat > mavg_hw.kg <<'EOF'
> mavg::.hw("examples/embedded/mavg.kg")
> .p(mavg.step([4.0 8.0 12.0 16.0 20.0]))
> .p(mavg.step(4.0))
> .hw.reset(mavg.step)
> .p(mavg.step(4.0))
> .p(.hw.info()@"mode")
> EOF
$ KLONG_HW=sim kgpy mavg_hw.kg
[1.0 3.0 6.0 10.0 14.0]
13.0
1.0
sim
```

The first call after `.hw` starts the program from its initial state; the five ticks are one run. The next call goes on from the window the run left. `.hw.reset(k)` makes the next call of `k` start from the initial state again.

## On the card

The same program with the card in place of the simulator. `KLONG_HW_HOST=user@host` names the machine with the card (key-based ssh; nothing listens on a socket). This is the session that was run against the image of `mavg.kg`, `fir.kg` and a small program of two traps, built together:

<!-- check-docs: skip: needs the Alveo card -->
```console
$ KLONG_HW_HOST=user@fpga-host kgpy card.kg
[3 241]
1
[1.0 3.0 6.0 10.0 14.0]
13.0
1.0
<hw buffer 9: 3 call(s) of fir.run, 3072 bytes>
[3 241]
1
card
fir_run_kernel, mavg_step_kernel, trap_f_kernel, trap_g_kernel
:{[report _x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt] [met 1] [wns_ns 0.027] [tns_ns 0.0] [whs_ns 0.009] [failing_endpoints 0]}
```

where `card.kg` is the program above with `.hw` of `fir.kg` and `mavg.kg`, `b::.hw.put(fir.run;x)`, `fir.run(b)`, `.hw.free(b)` and `.hw.info()` added. Binding `fir.kg` alone finds the image that was built with other programs, because the image holds a kernel built from exactly that source.

## Calling a kernel

`h::.hw("fir.kg")` binds each entry twice: as `h@"run"` in the returned dictionary (`g::h@"run"; g(x)`), and as the name `fir.run`, the file name without `.kg`, then a dot, then the entry. A list of paths, `.hw(["mavg.kg" "fir.kg"])`, binds the programs of an image built from several files.

* A call checks the number, type and shape of its arguments against the entry's signature before anything is sent, with the errors `call_entry` gives: a wrong type is a `type` error, a wrong shape a `shape` error, a wrong count an `arity` error, an `int` outside the kernel's 32 bits a `range` error.
* An argument with one extra leading dimension `n` is `n` calls in one run, and the result has the same leading dimension. All arguments of a call must agree on `n`.
* A float result is returned as float64, an int as int64, in the array type of the session's backend (a torch tensor on the torch backend). The arrays cross to the card as numpy.
* `.hw.put(k;x)` uploads the arguments of calls of `k` once and returns a buffer. For a kernel with several arguments `x` is a list of them. `k(b)` runs the stored calls without sending them again, `.hw.get(b)` downloads them and `.hw.free(b)` frees them. Buffers are freed when another image is loaded and when the session ends.
* `.hw.lanes("fir.kg";n)` is `.hw` for the image built with `--lanes n`. Every parameter of the program (declared with `.param`) is then an array of n values, one per lane, set with `.hw.set(h;"name";values)` between runs, and a result has a leading dimension n. `.hw.set` converts the value as `klongpy.static.set_param` does for the interpreter, and an unknown name, a wrong shape (one value per lane) or a value that does not fit the register is an error. See [Strategies](#strategies-many-parameter-sets-over-one-tick-history).
* `.hw.info()` is a dictionary: `mode` (`card`, `local` or `sim`), `device`, `shell`, the loaded `image` and its `key`, the `kernels`, the `build` record (routed timing, utilisation, build time), and the number and size of the `buffers`.

Loading an image replaces whatever the card held, and one session holds one image: a second `.hw` that needs another image loads it, and the kernels bound earlier then raise an error that says to call `.hw` again. Bind everything that is used together in one image.

### Errors

All are `system` errors (a subclass of `KlongSystemError`) except the first, which is the checker's own, and the last two rows, which are `name` and `range` errors.

| Situation | What you see |
|---|---|
| The program is outside the profile, or an entry cannot be a kernel | a `static` error with the position and the rule |
| No hardware: none of `KLONG_HW=sim`, `KLONG_HW_HOST`, `pyxrt` | a `system` error naming the three ways to get one |
| No image for this source, or the build failed timing, or it is still being built | a `system` error with the exact `kgpy --hw-build ...` command |
| The server refused an image (timing, no passing `build.json`, hash mismatch) | a `system` error; nothing is loaded |
| The card is in use by another process, or a unit is not idle | a `system` error; nothing is loaded |
| The connection dropped or the server stopped answering | a `system` error; call `.hw` again to reconnect |
| `.hw.set` of a name the program does not declare with `.param` | a `name` error that lists the parameters |
| A trap during a run: an index out of range, an integer division by zero | a `range` error naming the entry; the results of that run are not returned, and the next run starts clean |

The trap is the compiled program's trap: the kernel does not stop, it sets a flag, and the host raises after the run. The interpreter raises at the first call that does it.

<!-- check-docs: link examples -->
```console
$ cat > no_hw.kg <<'EOF'
> .hw("examples/embedded/fir.kg")
> EOF
$ env -u KLONG_HW -u KLONG_HW_HOST kgpy no_hw.kg
no_hw.kg:1:1: system error: no hardware to run on: set KLONG_HW_HOST=user@host to use the Alveo card on that machine over ssh, run KlongPy on the machine with the card (pyxrt importable), or set KLONG_HW=sim for the simulator (no hardware)
  .hw("examples/embedded/fir.kg")
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

## Strategies: many parameter sets over one tick history

`examples/hardware/crossover.kg` is a moving-average crossover strategy on one instrument. A tick of the price, as a
`fixed<32,16>`, goes in; the program keeps the sums of the last `fast` and `slow` prices (windows of up to 32 ticks, set
by parameters), a position in {-1, 0, 1}, the price it was taken at, a stop distance, the running profit and the largest
drawdown. The point of the hardware is to run *many parameter sets over one history*: with `--lanes 32` the kernel holds
32 copies of the strategy, each with its own `fast`, `slow` and `stop`, all fed by the same stream of ticks.

```
.param("fast";"int";5)                            :" the fast window, 1 to 32 ticks (outside: the nearest end) "
.param("slow";"int";20)
.param("stop";"fixed<32,16>";1)                   :" close a position that is this far under water "

step::{...update the account...;0}                .entry("step";"fixed<32,16> -> void")
summary::{profit,maxdd,wide(trades)}              .entry("summary";"-> fixed<64,32>[3]")
```

Four things make it fit the hardware.

* **A result per tick would cost more than the computation.** With 32 lanes a result of three 64-bit words per tick is
  96 words, and a kernel writes one word per clock cycle, so the store would be 96 times slower than the arithmetic.
  `step` therefore returns `void`, and `summary` reads the state at the end: the pair is one kernel that runs n calls of
  `step`, writes nothing for them, then writes the one record of `summary` (per lane, lane first: `[lanes 3]`). See
  [A summary entry](static-profile.md#a-summary-entry). `crossover.step(ticks)` returns the summary after those ticks,
  and `crossover.summary()` returns it for the state as it is, without running a tick.
* **There is no multiplication in what a tick carries over to the next.** A multiplication of two `fixed<32,16>` numbers
  in the feedback of a state variable limited the clock of a one-lane exponential moving average to 228 MHz (a C synthesis estimate). Here the
  sums change by an add and a subtract, the position by comparisons and the profit by an add; the two products of the
  crossover test (`fast sum * slow length` against `slow sum * fast length`) are computed from the sums and only feed the
  position, so the tool pipelines them. The position logic is written so that the comparison of the price with the stop
  is the only operation in its feedback loop before a final selection.
* **Each lane is a process of its own.** The kernel of a summary pair with lanes is not one loop that holds every lane. A
  single pipelined loop stalls on one condition, the input FIFO being empty, and that one signal drives the clock enable of
  every register of every lane: tens of thousands of loads on a net that no register drives, which cannot be replicated
  or pipelined and routes at several times the clock period. With one dataflow process per lane (the program's C is
  compiled once per lane under other names, `lane_<l>.c`, so that a lane owns its instance, its functions and its trap flag,
  as a process may not share a variable with another), the stall net of a lane reaches a few thousand registers. The
  input words are copied to every lane's stream; the records of the lanes are collected in lane order at the end.
* **A parameter is an array of one value per lane.** `.hw.lanes("crossover.kg";32)` binds the 32-lane image, and
  `.hw.set(h;"fast";values)` sets a parameter between runs, converted as the interpreter converts it
  (`klongpy.static.set_param`): an int for `fast`, a number for the `fixed<32,16>` `stop`, truncating toward minus infinity.
  The kernel keeps the value, through resets, until it is set again. The values of every parameter are read by the
  kernel from a table in card memory when it starts (the host runtime can write a scalar to a kernel's register but not
  an array), which the server fills from the values it keeps; a program without lanes has its parameters as scalar
  registers of the kernel.

On the simulator, with two lanes (`fast` 2 and 1, `slow` 4 and 3) and the 20 ticks that the tests check by hand:

<!-- check-docs: link examples; requires-cmd cc c++ -->
```console
$ cat > strategy.kg <<'EOF'
> s::.hw.lanes("examples/hardware/crossover.kg";2)
> .hw.set(s;"fast";[2 1])
> .hw.set(s;"slow";[4 3])
> .hw.set(s;"stop";[1.5 1.5])
> t::[10 10 10 10 11 9.25 9.5 9.75 11.5 12 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17]
> .p(crossover.step(t))
> .p(crossover.summary())
> EOF
$ KLONG_HW=sim kgpy strategy.kg
[[1.75 3.75 3.0] [5.0 2.25 3.0]]
[[1.75 3.75 3.0] [5.0 2.25 3.0]]
```

Each row is one lane: profit, largest drawdown, number of trades. The first lane is the series worked by hand in the test
(`tests/static/test_crossover.py`); the same account comes out of the interpreter stand-in, the emitted C, the kernel in
C simulation and this simulator, bit for bit.

Building and replaying it on the card:

```
kgpy --hw-build examples/hardware/crossover.kg --lanes 32 --clock-mhz 250
python examples/hardware/crossover_bench.py gen history.bin params.bin --ticks 10000000 --lanes 32
KLONG_HW_HOST=user@host python examples/hardware/crossover_card.py history.bin params.bin
```

## Where the hardware runs

| Setting | Hardware used |
|---|---|
| `KLONG_HW=sim` | the simulator, on this machine; no card, no image |
| `KLONG_HW_HOST=user@host` | the card on that machine; the server `klongpy/hw/server.py` is started there over ssh with Python 3.8, numpy and pyxrt |
| neither, and `pyxrt` importable | the card on this machine |
| `KLONG_HW_BUILD=user@host` | where `--emit-hls --synth` and `--hw-build` run |

The build machine and the card machine may be the same machine but are separate settings. Nothing of this is imported when KlongPy starts.

These settings tell KlongPy where the tools and its files are on the build machine. They are read where `kgpy` runs and used on the machine named by `KLONG_HW_BUILD` or `KLONG_HW_HOST`:

| Setting | Meaning | Default |
|---|---|---|
| `KLONG_HW_ROOT` | the directory on the build and card machine under which everything is written (`work/`, `builds/`, `server/`, `venv/`, `build.lock`); an absolute path, or one that starts with `~/`, which is the home directory of that machine; a path outside it is refused | `~/klong-hw`, found on that machine |
| `KLONG_HW_VITIS_SETTINGS` | the `settings64.sh` of Vitis that a build sources before `v++` and `vitis_hls` | none: `v++` and `vitis_hls` must be on the `PATH` of the machine, or the build stops and names this variable |
| `KLONG_HW_HLS_SETTINGS` | the `settings64.sh` of Vitis HLS that `--synth` and the remote tests source before `vitis_hls` | none: `vitis_hls` must be on the `PATH` of the machine |
| `KLONG_HW_PLATFORM_REPO_PATHS` | the directory that holds the U250 platform (`xilinx_u250_gen3x16_xdma_4_1_202210_1`), exported to the build as `PLATFORM_REPO_PATHS` | none: a build stops with an error naming this variable unless `PLATFORM_REPO_PATHS` is already set on the machine |

The hash that names a build directory does not include the root, so images built under one root are found again when the root is set to that directory.

### Trust

The server runs on the card machine with your ssh identity, and the build runs there as you. Loading an image replaces what was on the card, including an image of someone else's, after checking that no other process has the card open. The loader does not program the base shell and never loads an image whose timing failed. All of its files are under `KLONG_HW_ROOT` (default `~/klong-hw/`) on that machine. Nothing is installed outside it, and nothing needs sudo.

## Measured

What was measured, with its tier. These are hardware numbers on an Alveo U250 for the `fir` image above (built 2026-11-16, 300 MHz requested and met, worst slack 0.027 ns), with numpy measured on the same programs on the Mac that ran KlongPy and on the CPU of the card machine. Best of 3 to 5 runs (2 for 100000 blocks). `put`, `kernel` and `get` are the server's clock on the card machine, `card wall` is what the client saw including ssh from the Mac.

| blocks | put (s) | kernel (s) | get (s) | card wall (s) | numpy Mac (s) | numpy card CPU (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000045 | 0.000084 | 0.000015 | 0.002701 | 0.000011 | 0.000015 |
| 1000 | 0.001590 | 0.013105 | 0.001688 | 0.040323 | 0.004779 | 0.005158 |
| 100000 | 0.144330 | 1.294230 | 0.158533 | 3.400952 | 1.025725 | 1.046810 |

The kernel takes about 12.9 microseconds per block (about 3886 cycles at 300 MHz). That is the first stage of the flow, in which the hardware runs the program's C unchanged and cannot do a floating-point sum at one result per cycle; numpy takes about 10 microseconds per block on the same gather and weighted sum (not an optimised convolution). The card is slower than numpy at 1 block (the overhead dominates), at 1000 blocks, and at 100000 blocks even before the transfer and the ssh round trip are counted. No crossover exists for this kernel. A kernel that is rebuilt as a shift register with a pipelined reduction is where speed would have to come from; that is not built.

Image of `mavg.kg`, `fir.kg` and the trap program: 126578 LUT (7.33 percent of the device with the shell), 199444 FF, 276.5 BRAM tiles, 18 DSP.

### Measured: the strategy kernel on the card

Hardware tier, an Alveo U250 in a workstation (Ryzen 9 7950X3D, 16 cores and 32 threads, gcc 9.4, numpy 1.24 for the host side), measured 2026-11-17. The image is `crossover.kg` built with `--lanes 32 --clock-mhz 250`: it passed routed timing at 250 MHz (worst setup slack 0.009 ns, worst hold slack 0.009 ns, no failing endpoints; the margin is thin) after a build of 1 hour 54 minutes, and uses 13.6 percent of the device's LUTs (234315, the shell's 118282 included), 7.0 percent of its registers, 258 block RAM tiles and 135 DSPs. The history is 10 million synthetic ticks, a random walk with rare jumps folded into the band 5 to 15 (`crossover_bench.py gen`), held as a file on the machine next to the card; the 32 parameter sets are 4 values of `fast`, 4 of `slow` and 2 stops (`param_grid`). Every lane replays all 10 million ticks, 320 million lane-ticks in all.

| | time for 10 M ticks x 32 lanes | lane-ticks per second, total | ticks per second per lane |
|---|---:|---:|---:|
| card, kernel only (the server's clock, submission to completion; best of 7) | 40.2 ms | 7.97 G | 249 M |
| card, history resident on the card, wall time of the call from the workstation over ssh (best of 7) | 42.7 ms | 7.50 G | 234 M |
| card, the same with the 40 MB history uploaded first (from the workstation over ssh) | 0.56 s | 0.57 G | 18 M |
| the emitted C, `gcc -O3`, one core, the 32 lanes in turn (best and worst of 5) | 6.0 s to 6.8 s | 47 M to 53 M | 47 M to 53 M |
| the same, 8 processes of 4 lanes | 0.87 s | 366 M | 46 M |
| the same, 16 processes of 2 lanes (one per core; best of 5) | 0.46 s | 691 M | 43 M |
| the same, 32 processes of 1 lane (one per thread; best of 5) | 0.42 s | 758 M | 31 M |
| numpy (`numpy_run`: the sums and the signal for all lanes at once, the position stepped one tick at a time), measured on 1 M ticks | 20.3 s for 1 M ticks | 1.6 M | 50 k |

The card replays one tick per clock cycle in every lane (250 M a second at 250 MHz), 32 lanes at once. Against the C that the same Klong source compiles to, it is 10 times faster than all 32 threads of the CPU (kernel 40.2 ms against 0.42 s) and 150 times faster than one core, and a lane on the card is five times a core. The end-to-end figure, 42.7 ms, includes the ssh round trip from the workstation (about 2 ms); the upload of the history is a separate cost, and it is the largest one when the data has to come over ssh (0.52 s for 40 MB), against 0.08 s for the transfer from the machine's memory to the card that the server's clock measured. The CPU's numbers include reading the history from the page cache. The numpy version is not a fair opponent for long histories: the position of a tick depends on the position of the one before, so it steps one tick at a time (50 thousand ticks a second for all lanes); only the sums and the signal vectorise.

Where the CPU wins. The card has a fixed cost per call of about 1 ms (its kernel takes 0.05 ms for 1000 ticks, a call from the workstation 1.0 ms), so a computation of less than about 50 thousand ticks of one parameter set is faster on one core (one lane: 50 M ticks a second, 20 ns per tick); the card needs all of its 32 lanes to be worth it, and with 32 parameter sets it already wins at 1000 ticks (2.4 ms on one core, 1.0 ms on the card) and by 12 times at 100000 ticks (card 1.7 ms against 21 ms for 16 processes). The number of lanes and every `.entry` are fixed at build time and a build takes hours; a CPU compiles the program in a second and runs any number of lanes. The first build of this kernel, with 64 lanes in one pipelined loop, failed timing before routing (see below), and the working design has half the lanes.

| ticks | card kernel | card wall (workstation) | CPU, 1 core | CPU, 16 processes | CPU, 32 processes |
|---:|---:|---:|---:|---:|---:|
| 1 000 | 0.05 ms | 1.0 ms | 2.4 ms | 15.7 ms | 32 ms |
| 10 000 | 0.09 ms | 1.1 ms | 6.9 ms | 16.3 ms | 31 ms |
| 100 000 | 0.45 ms | 1.7 ms | 60 ms | 20.8 ms | 32 ms |
| 1 000 000 | 4.1 ms | 5.9 ms | 0.59 s | 63 ms | 67 ms |
| 10 000 000 | 40.2 ms | 43 ms | 5.9 s | 0.47 s | 0.43 s |

(best of 9 calls for the card, of 5 for the CPU; 32 lanes; the CPU's processes start together and each reads its ticks, so the small cases show the cost of starting 16 or 32 processes.)

Correctness. The summary of every lane after the first 100000 ticks equals the interpreter stand-in's (`klongpy.static.eval_ir`, which `call_entry` uses), and after all 10 million ticks equals the emitted C's, bit for bit, on the card. The hand-checked 20-tick series give the worked results in every lane, and the numpy version equals the interpreter. `tests/static/test_crossover_card.py` runs the hand series and 200000 ticks against the card.

## What has not been done

* One image has been run on the card for the strategy kernel, and the earlier image of `mavg.kg`, `fir.kg` and the trap program has not been run since the emitter changed (its key no longer matches the source, so `tests/static/test_hw_card.py` and `test_hw_binding_card.py` skip until it is built again: `kgpy --hw-build examples/embedded/mavg.kg examples/embedded/fir.kg tests/static/hw/trap.kg`).
* More than 32 lanes: a first build with 64 lanes in one pipelined loop did not meet timing (estimated worst slack -6.4 ns at 250 MHz after placement, with five nets of more than 10000 loads; it was stopped during routing), and 64 lanes as one process per lane (about 290000 LUTs) has not been built.
* No hardware emulation result: loading the emulated image did not return within the time given. No co-simulation.
* No network ports. An entry that mixes `int` and `float` arguments is built but cannot be called by `.hw` yet, and a summary kernel with lanes needs records of one width.
* On the numbers above, "working on hardware" means this image, which passed routed timing, produced the interpreter's answers on the card (the strategy kernel exactly; floats within 1e-5 relative for `fir`).
