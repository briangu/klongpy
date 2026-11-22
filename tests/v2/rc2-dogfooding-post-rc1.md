# RC2 dogfooding notes after the RC1 source freeze

## Checking a function supplied by a board

The P4 diagnostic's C function fills 15 integers. In Klong, `r::.c.check()` returns three five-element rows. An initial test indexed the outer list as a 15-element vector and hit `index 4 out of range for list of 3`. The host console driver continued to the following line and exited successfully. Running the whole file through `kgupload.py PORT run check-compiled.kg` gives the caller a nonzero exit status on a Klong error.

The release check now requires `[3 5]~^r` before it flattens the rows with `,/r`. It checks the interpreter timing for each example as well as the pass flag. At a 92,000-byte heap the FIR row can pass against its pinned reference while reporting an interpreter time of −1. The P4 check rejected that result on the host. A wide-heap execution of the same C functions passed for all 64 moving-average, 32 PID and 241 FIR outputs.

Desktop `kg lint` warns that `.c.check` is undefined in the file. For RC2, a useful check for a board profile would recognize the names its firmware registers and still catch a misspelling. The stock simulator returned a name error for `.c.check()`, and the upload client propagated it with exit status 1.

## First launch on an Intel Mac wheel under Rosetta

The first `kg --help` after installing the original Intel Mac wheel took 10.904 seconds on Apple Silicon and exceeded the agent test's ten-second timeout. The final RC1 wheel passed on first launch. A Klong script that created a native table and queried its sum also finished within the deadline on a fresh executable. [Measured startup and warm query results](../../docs/performance.md#first-launch-under-rosetta)
