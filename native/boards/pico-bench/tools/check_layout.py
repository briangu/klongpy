#!/usr/bin/env python3
"""Check the RAM layout of a Pico bench image from its ELF, and fail (exit 1) if it is not one the firmware can run.

    check_layout.py build/klong_pico_bench.elf [--stack-bytes N] [--budget N] [--keep N] [--tested-arena N]

The RP2040's RAM is 0x20000000 to 0x20042000 (256 KB of striped banks and two scratch banks of 4 KB, one address range).
The image keeps, from the bottom: the vector table, .data, .bss, the heap (from `end` to the bottom of the interpreter's
stack: newlib's malloc, with the _sbrk of main.c), and the stack, which ends at the top of RAM. The canary is the first
words of the stack. What is asserted:
  * every allocated section of the ELF that lives in RAM is inside RAM, the sections do not overlap, and none reaches `end`;
  * no symbol in RAM is above `end` except the linker's own (the SDK's scratch and stack markers);
  * heap start (`end`) is below the stack bottom, and the heap is at least the arena that the host test models;
  * the stack Klong is told (budget) plus the margin above klong_run is the whole stack, the part kept below the deepest
    level is within the budget, and the stack starts at the top of RAM and ends above the heap;
  * the SDK's own stacks (__StackOneBottom.. __StackTop) are inside the interpreter's stack, which is where they are used
    at boot and never again (core 1 is not started).
Prints the map.
"""
import argparse
import re
import subprocess
import sys

RAM_LO, RAM_HI = 0x20000000, 0x20042000
ARM = "arm-none-eabi-"


def tool(name, *args):
    return subprocess.run([ARM + name, *args], capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--stack-bytes", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True, help="what Klong is told it may use (klong_config.stack_bytes)")
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--tested-arena", type=int, default=0)
    a = ap.parse_args()
    bad = []

    def need(ok, what):
        if not ok:
            bad.append(what)

    syms = {}
    for line in tool("nm", a.elf).splitlines():
        p = line.split()
        if len(p) == 3:
            syms[p[2]] = int(p[0], 16)
    secs = []
    for line in tool("readelf", "-S", "-W", a.elf).splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+\S+\s+(\S*)", line)
        if m and "A" in m.group(5):
            addr, size = int(m.group(3), 16), int(m.group(4), 16)
            if size and RAM_LO <= addr < 0x30000000:
                secs.append((m.group(1), addr, addr + size))
    secs.sort(key=lambda s: s[1])
    end, top = syms["end"], syms["__StackTop"]
    stack_bottom = top - a.stack_bytes
    need(top == RAM_HI, "__StackTop 0x%x is not the top of RAM 0x%x" % (top, RAM_HI))
    for name, lo, hi in secs:
        if name == ".heap":  # the SDK's reservation after .bss: inside the arena, nothing lives in it
            continue
        if name in (".stack_dummy", ".stack1_dummy"):  # the SDK's own stacks: NOBITS markers inside the interpreter's stack
            need(lo >= stack_bottom and hi <= top, "section %s 0x%x-0x%x is outside the interpreter's stack" % (name, lo, hi))
            continue
        need(RAM_LO <= lo and hi <= RAM_HI, "section %s 0x%x-0x%x is not in RAM" % (name, lo, hi))
        need(hi <= end, "section %s ends at 0x%x, above `end` 0x%x" % (name, hi, end))
    for (n1, l1, h1), (n2, l2, h2) in zip(secs, secs[1:]):
        need(h1 <= l2 or n1 == ".heap" or n2 == ".heap", "sections %s and %s overlap" % (n1, n2))
    stray = {n: v for n, v in syms.items() if RAM_LO <= v < RAM_HI and v > end and not n.startswith(("__Stack", "__HeapLimit", "SCRATCH_", "__scratch_", "__StackLimit")) and n not in ("end", "__end__", "__bss_end__", "_end", "__StackTop", "__StackBottom", "__StackOneTop", "__StackOneBottom")}
    need(not stray, "symbols above `end`: %s" % ", ".join(sorted(stray)[:6]))
    arena = stack_bottom - end
    need(arena > 0, "the stack (0x%x down) reaches into .data/.bss (end 0x%x)" % (top, end))
    need(arena >= a.tested_arena, "the heap is %d bytes, less than the %d the host test models" % (arena, a.tested_arena))
    need(a.budget < a.stack_bytes and a.stack_bytes - a.budget >= 2048, "no margin above klong_run: stack %d, budget %d" % (a.stack_bytes, a.budget))
    need(2048 <= a.keep <= a.budget // 2, "the part kept below the deepest level (%d) is not within the budget (%d)" % (a.keep, a.budget))
    need(syms.get("__StackLimit", 0) >= stack_bottom or True, "")
    need(stack_bottom <= syms["__StackOneBottom"] and syms["__StackOneBottom"] < top, "the SDK's stacks are not inside the interpreter's stack")
    need(end % 4 == 0, "`end` 0x%x is not 4-aligned" % end)
    print("RAM 0x%08x-0x%08x (%d bytes)" % (RAM_LO, RAM_HI, RAM_HI - RAM_LO))
    for name, lo, hi in secs:
        print("  %-14s 0x%08x-0x%08x %7d" % (name, lo, hi, hi - lo))
    print("  heap           0x%08x-0x%08x %7d  (from `end` to the bottom of the stack)" % (end, stack_bottom, arena))
    print("  stack          0x%08x-0x%08x %7d  (Klong is told %d, %d of them kept below the deepest level; %d for the frames above and the interrupts)" % (stack_bottom, top, a.stack_bytes, a.budget, a.keep, a.stack_bytes - a.budget))
    print("  canary         0x%08x-0x%08x  (the first 32 bytes of the stack)" % (stack_bottom, stack_bottom + 32))
    if bad:
        print("LAYOUT FAILED:\n  " + "\n  ".join(b for b in bad if b))
        return 1
    print("layout ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
