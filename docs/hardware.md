# Klong functions as FPGA hardware

!!! warning "Experimental"
    This is new and has been run on one machine and one card, an Alveo U250. The commands, the names `.hw` and `--hw-build`, and the file formats may change. In the first version the hardware does not beat numpy on the one kernel that was measured (see [Measured](#measured)); what it gives you is the path itself: one Klong source that runs interpreted, as C, and on the card, with the same answers.

A function of the [static profile](static-profile.md) can be built into a hardware kernel and called from a KlongPy session with ordinary arrays. The same source runs three ways: interpreted, as C (see [Compiling Klong to C](embedded.md)), and on the card.

```
kgpy --emit-hls fir.kg -o out/       # seconds: HLS C++ and build scripts
kgpy --hw-build fir.kg               # hours: a bitstream on the build machine, cached by source
fir::.hw("fir.kg")                   # in KlongPy: bind the built kernels
y::fir.run(x)                        # runs on the card
```

## What qualifies

A program is a candidate when it is inside the static profile (`kgpy --check-static FILE`) and an `.entry` of it is a function that

* takes at least one argument and returns a value (a record in, a record out),
* takes `int` and `float` arguments and arrays of them with fixed shapes, not `str`,
* does not reach an `.extern`: the card has nothing to call,
* does not mix `int` and `float` arguments in one entry, for the host side (`.hw` refuses such an entry; the build accepts it),
* in a float program computes in 32-bit `float` (the card has no `--float64`); `int` is 32 bits unless the program was built with `--int64`.

Every `.entry` becomes one kernel. The state of the program (its globals) is kept inside the kernel between calls. Two entries of one program are two kernels with a state each; they do not share globals.

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
$ KLONG_HW_HOST=brian@xilinx kgpy card.kg
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
* `.hw.info()` is a dictionary: `mode` (`card`, `local` or `sim`), `device`, `shell`, the loaded `image` and its `key`, the `kernels`, the `build` record (routed timing, utilisation, build time), and the number and size of the `buffers`.

Loading an image replaces whatever the card held, and one session holds one image: a second `.hw` that needs another image loads it, and the kernels bound earlier then raise an error that says to call `.hw` again. Bind everything that is used together in one image.

### Errors

All are `system` errors (a subclass of `KlongSystemError`) except the first, which is the checker's own.

| Situation | What you see |
|---|---|
| The program is outside the profile, or an entry cannot be a kernel | a `static` error with the position and the rule |
| No hardware: none of `KLONG_HW=sim`, `KLONG_HW_HOST`, `pyxrt` | a `system` error naming the three ways to get one |
| No image for this source, or the build failed timing, or it is still being built | a `system` error with the exact `kgpy --hw-build ...` command |
| The server refused an image (timing, no passing `build.json`, hash mismatch) | a `system` error; nothing is loaded |
| The card is in use by another process, or a unit is not idle | a `system` error; nothing is loaded |
| The connection dropped or the server stopped answering | a `system` error; call `.hw` again to reconnect |
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

## Where the hardware runs

| Setting | Hardware used |
|---|---|
| `KLONG_HW=sim` | the simulator, on this machine; no card, no image |
| `KLONG_HW_HOST=user@host` | the card on that machine; the server `klongpy/hw/server.py` is started there over ssh with Python 3.8, numpy and pyxrt |
| neither, and `pyxrt` importable | the card on this machine |
| `KLONG_HW_BUILD=user@host` | where `--emit-hls --synth` and `--hw-build` run |

The build machine and the card machine are the same machine today but are separate settings. Nothing of this is imported when KlongPy starts.

### Trust

The server runs on the card machine with your ssh identity, and the build runs there as you. Loading an image replaces what was on the card, including an image of someone else's, after checking that no other process has the card open. The loader does not program the base shell and never loads an image whose timing failed. All of its files are under `/home/<user>/klong-hw/` on that machine. Nothing is installed outside it, and nothing needs sudo.

## Measured

What was measured, with its tier. These are hardware numbers on an Alveo U250 for the `fir` image above (built 2026-11-16, 300 MHz requested and met, worst slack 0.027 ns), with numpy measured on the same programs on the Mac that ran KlongPy and on the CPU of the card machine. Best of 3 to 5 runs (2 for 100000 blocks). `put`, `kernel` and `get` are the server's clock on the card machine, `card wall` is what the client saw including ssh from the Mac.

| blocks | put (s) | kernel (s) | get (s) | card wall (s) | numpy Mac (s) | numpy card CPU (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000045 | 0.000084 | 0.000015 | 0.002701 | 0.000011 | 0.000015 |
| 1000 | 0.001590 | 0.013105 | 0.001688 | 0.040323 | 0.004779 | 0.005158 |
| 100000 | 0.144330 | 1.294230 | 0.158533 | 3.400952 | 1.025725 | 1.046810 |

The kernel takes about 12.9 microseconds per block (about 3886 cycles at 300 MHz). That is the first stage of the flow, in which the hardware runs the program's C unchanged and cannot do a floating-point sum at one result per cycle; numpy takes about 10 microseconds per block on the same gather and weighted sum (not an optimised convolution). The card is slower than numpy at 1 block (the overhead dominates), at 1000 blocks, and at 100000 blocks even before the transfer and the ssh round trip are counted. No crossover exists for this kernel. A kernel that is rebuilt as a shift register with a pipelined reduction is where speed would have to come from; that is not built.

Image of `mavg.kg`, `fir.kg` and the trap program: 126578 LUT (7.33 percent of the device with the shell), 199444 FF, 276.5 BRAM tiles, 18 DSP.

## What has not been done

* Any image other than that one has not been run on the card. No fixed-point types, no `--lanes`, no network ports.
* No hardware emulation result: loading the emulated image did not return within the time given.
* The first stage is the program's C unchanged. A streaming core, which is where speed would come from, is not built.
* An entry that mixes `int` and `float` arguments is built but cannot be called by `.hw` yet.
* On the numbers above, "working on hardware" means this image, which passed routed timing, produced the interpreter's answers on the card (float results within 1e-5 relative, integer results exactly).
