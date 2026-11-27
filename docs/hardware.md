# Klong functions as FPGA hardware

A function of the [static profile](static-profile.md) becomes an FPGA kernel, and you call it from a KlongPy session with ordinary arrays. This page is for someone who has an FPGA card, or wants to know what the flow produces and how fast it is.

!!! warning "Experimental"
    This has been run on one card, an Alveo U250. The commands, the names `.fpga` and `kgpy fpga build`, and the file formats may change. The strategy kernel keeps a tick history on the card for repeated runs. The FIR filter returns nearly as much data as it consumes. The [measurements](#results) show kernel time, transfer time and CPU comparisons for both workloads.

The same source runs three ways: interpreted, as C (see [Compiling Klong to C](embedded.md)) and on the card, with the same answers. Two kinds of kernel exist. A *streaming kernel*, the default, takes one record per clock cycle when the program allows it. A *wrapped kernel* (`--wrapped`) runs the program's C unchanged once per record and is slower.

`fpga source` is part of the engine and also runs under `kg`; building, the card server and `.fpga` are in `kgpy`. They need no Python packages beyond KlongPy's own. The machine with the card also needs the Xilinx runtime (`pyxrt`, which is not on PyPI), and the build machine needs Vitis.

## Tested setup

| | |
|---|---|
| Card | Alveo U250 |
| Platform | `xilinx_u250_gen3x16_xdma_4_1_202210_1` |
| Toolchain | Vitis and Vitis HLS 2022.2 |
| Host | Ryzen 9 7950X3D (16 cores, 32 threads), gcc 9.4, numpy 1.24 |
| Card link | PCIe Gen3 ×1 (8 GT/s per lane). The U250 supports Gen3 ×16. |
| Dates | Images built November 16–17, 2026. Board regression and offload measurements repeated November 27. |
| Images run | Wrapped FIR, streaming FIR with moving average and trap kernels, and the 32-lane streaming strategy |

### PCIe topology and timing boundaries

The benchmark client sent data over SSH to the Ryzen card host. On that host, XRT copied buffers between host memory and U250 memory over PCIe. Linux reported the following negotiated links on the path to the accelerator. The upstream and downstream ports belong to the same AMD PCIe bridge complex.

| Physical link, from host toward card | Negotiated link | What the ports advertise |
|---|---|---|
| AMD root port to bridge upstream port | Gen4 ×4 (16 GT/s per lane) | Upstream port supports Gen4 ×4 |
| Bridge downstream port to U250 | **Gen3 ×1 (8 GT/s)** | Downstream port supports Gen3 ×1. U250 endpoint supports Gen3 ×16. |

The `gen3x16` in the platform name describes the card platform. On this host the final hop has a theoretical ceiling of **985 MB/s in each direction** after Gen3 line encoding and before packet overhead (8 GT/s × 128/130 ÷ 8). The 100 GbE link in the ticker plant measurements connected two other hosts. [Recorded PCIe topology](https://github.com/briangu/klongpy/blob/main/examples/hardware/results/2026-11-27-u250.json) · [Ticker plant workload and hardware](performance.md#fpga-analytics-on-committed-trades)

The **upload and download** timers run on the card host and include XRT buffer handling, host copies and device synchronization. The **kernel** timer runs from submission to completion with inputs already in card memory. **Client wall time** includes the SSH round trip, client and server processing, and the card operations. The CPU comparison on the card host uses host memory. These experiments used one link width, so the results do not quantify the speedup from a wider slot.

To check another Linux host, take the accelerator's domain-qualified PCI address from `lspci -D`. Follow its sysfs path toward the root port to identify where the width changes. The command prints current speed and width followed by maximum speed and width for each device.

```sh
device=$(readlink -f "/sys/bus/pci/devices/${BDF:?Set BDF to the accelerator PCI address}")
while [ -r "$device/current_link_width" ]; do
    printf '%s  ' "${device##*/}"
    paste "$device/current_link_speed" "$device/current_link_width" \
          "$device/max_link_speed" "$device/max_link_width"
    device=$(dirname "$device")
done
```

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
(`tests/static/test_crossover.py`); the same account comes out of the evaluator of the static profile (the one `call_entry` uses), the emitted C, the kernel in
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

The kernel takes about 12.9 microseconds per block, about 3886 cycles at 300 MHz. A wrapped kernel cannot do a floating-point sum at one result per cycle. Numpy takes about 10 microseconds per block on the same gather and weighted sum. The card is slower than numpy at every size, even before the transfer is counted. The streaming image uses a shift register and a pipelined reduction.

The image of `mavg.kg`, `fir.kg` and `tests/static/hw/trap.kg` (a program that traps on purpose) uses 126578 LUT (7.33 percent of the device with the shell), 199444 FF, 276.5 BRAM tiles and 18 DSP.

### Streaming FIR on the U250

On November 27, 2026, the current source matched a cached 300 MHz image with 0.055 ns worst setup slack and zero failing endpoints. The physical-board regression compared 1,000 blocks with the interpreter. The table shows the trial with the shortest client wall time, including upload and the return of all 241 outputs per block.

| Blocks | Upload on card host | Kernel | Download on card host | Client wall over SSH | NumPy on card host |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.063 ms | 0.066 ms | 0.043 ms | 1.803 ms | 0.015 ms |
| 1,000 | 1.439 ms | 1.193 ms | 1.444 ms | 24.626 ms | 5.231 ms |
| 100,000 | 144.539 ms | 105.657 ms | 161.983 ms | 2,229.462 ms | 953.694 ms |

At 100,000 blocks the streaming kernel took 1.06 µs per block. The earlier wrapped kernel took 12.94 µs. The streaming run transferred 102.4 MB of inputs and returned 96.4 MB of results over the SSH connection and a Gen3 ×1 PCIe link. NumPy on the card host finished the complete job about 2.3 times faster than the remote call. To repeat the comparison, run `python -m klongpy.hw.bench --numpy-host --sizes 1,1000,100000` with the card host configured.

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

### Reusing one history across 1,024 strategies

On November 27 the 32-lane, 250 MHz kernel replayed 10 million prices in 40.23 ms median across nine runs. The median call over SSH was 42.71 ms with prices in card memory. All 32 final summaries matched the emitted C, and the one-million-tick prefix matched NumPy.

A second experiment ran 1,024 distinct configurations on that same image. Each of 32 calls set new fast windows, slow windows and stop distances, reset all 32 accounts and reused the same 40 MB input. Each configuration returned three numbers. Across three command invocations, every one of the 3,072 outputs matched the emitted C bit for bit.

| Stage | Measured time |
|---|---:|
| Bind the image | 1.20–1.36 s |
| Upload the history once via `.fpga.put` | 0.50–0.55 s |
| Run all 32 groups and write their summaries | 1.46–1.49 s |
| Complete process, from launch through exit | 3.42–3.65 s |

That is 10.24 billion **strategy updates** per sweep. The input consists of 10 million distinct time steps shared by all strategies. The sweep counts parameter changes and result retrieval. The complete process also counts reading the input files and syncing the 24 KiB output. On the card host, 32 processes running the same emitted C with GCC `-O3` completed the sweep in 11.19–11.30 s across three trials. The process launches and reading the history from the page cache are included in the CPU times. [Trial timings, image hashes and correctness checks](https://github.com/briangu/klongpy/blob/main/examples/hardware/results/2026-11-27-u250.json)

To reproduce from the checkout root, generate the history with `python examples/hardware/crossover_bench.py gen history.bin baseline-params.bin --ticks 10000000`. Save and run the following with desktop `kg` to generate the 1,024 triples. Use fresh output paths for both scripts.

```klong
param::{[;1+x:%128;9+(x:%8)!16;256*1+x!8]}
f::.bfile("params1024.bin";:create)
f([;:write;4;,/param'!1024]);f(:sync)
```

In a card-enabled `kgpy` session, run the [Klong sweep](https://github.com/briangu/klongpy/blob/main/examples/hardware/crossover_sweep.kg) with these paths.

```klong
paths::["examples/hardware/crossover.kg" "history.bin" "params1024.bin" "summary.bin"]
.l("examples/hardware/crossover_sweep.kg")
```

To check the output independently, use the same history and parameter file with the emitted C.

```sh
kg compile c examples/hardware/crossover.kg -o c/
python examples/hardware/crossover_bench.py cpu c/ history.bin params1024.bin --procs 32 --out cpu-summary.bin
cmp summary.bin cpu-summary.bin
```

### A trade bar in C simulation and synthesis

The [trade-bar example](https://github.com/briangu/klongpy/blob/main/examples/hardware/trade_bar.kg) holds one symbol and one interval per reset. For each ordered trade it takes a price in integer ticks and a quantity in lots. It returns open, high, low, close, total quantity, the sum of price × quantity, and trade count. The host chooses the symbol and interval, and can divide the weighted sum by total quantity to get VWAP. In the ticker plant, a subscriber could run this calculation on acknowledged batches while the head continues to own sequencing and disk sync. Choose units and a window that keep the sums within signed 64-bit range.

On November 27, Vitis C simulation processed 200,003 synthetic trades in chunks of 4,093. All seven outputs matched an independent integer reference, including the weighted sum of 587,990,033,025. With `--int64 --clock-mhz 250`, HLS estimated 10,505 LUTs and 10 DSPs. It scheduled one core update per cycle. The input port reads one 64-bit word per cycle and each trade needs two words. At the target clock that gives a design ceiling of 125 million trades/s before transfer overhead. This example has reached C synthesis. Physical timing and throughput remain unmeasured. [How a ticker plant subscriber could use the kernel](performance.md#fpga-analytics-on-committed-trades)

### Correctness

The summary of every lane after the first 100000 ticks equals the static profile's evaluator's (the one `call_entry` uses). After all 10 million ticks it equals the emitted C's, bit for bit, on the card. The hand-checked 20-tick series give the worked results in every lane. `tests/static/test_crossover_card.py` runs the hand series and 200000 ticks against the card. For `fir`, floats agree within 1e-5 relative.

The November 27 hardware run passed all 20 tests in `test_hw_card.py`, `test_hw_binding_card.py` and `test_crossover_card.py`. It exercised both current images, including state continuation, reset, resident buffers, runtime traps and image rejection.

## Known limits

* **64 physical lanes.** An earlier design with all lanes in one loop missed timing after placement. On November 27 the one-process-per-lane design completed C synthesis targeting 250 MHz. Vitis estimated 264,417 LUTs, 203,669 registers and 256 DSPs for the kernel. All 64 tick loops had an initiation interval of one clock. The 192 output words are written once per replay. A 10,000-tick, 64-configuration software simulation matched two passes on the physical 32-lane image. The 64-lane design still needs place-and-route and a board run.
* Hardware emulation: loading the emulated image did not complete in the time allowed. There is no co-simulation.
* There are no network ports.
* An entry that mixes `int` and `float` arguments is built but cannot be called by `.fpga`, and a summary kernel with lanes needs records of one width.
* One session holds one image. Loading an image replaces what the card held.
* Builds take hours and one image at a time.
