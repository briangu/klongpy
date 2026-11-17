#!/usr/bin/env python3
"""
Section sizes of an ELF image, and what its code is made of.

    python3 native/scripts/sections.py FILE [--top N] [--heap BYTES]

Prints .text, .rodata, .data and .bss (the probe's heap of --heap bytes is taken off .bss),
then, when the file has a symbol table, the size of the functions and constants grouped by
the crate they come from and the N largest symbols. Only the standard library is used
(no llvm-size, no cargo-bloat).
"""
import re
import struct
import sys


def read(path):
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"\x7fELF" or b[4] != 1 or b[5] != 1:
        raise SystemExit(f"{path}: not a 32-bit little-endian ELF file")
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    secs = []
    for i in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from("<10I", b, shoff + i * shentsize)
        secs.append(dict(name=name, type=typ, flags=flags, addr=addr, off=off, size=size, link=link, entsize=entsize))
    strtab = secs[shstrndx]

    def cstr(tab, at):
        end = b.index(b"\0", tab["off"] + at)
        return b[tab["off"] + at:end].decode("utf-8", "replace")

    for s in secs:
        s["name"] = cstr(strtab, s["name"])
    syms = []
    for s in secs:
        if s["type"] == 2 and s["entsize"]:  # SHT_SYMTAB
            names = secs[s["link"]]
            for k in range(s["size"] // s["entsize"]):
                nm, value, size, info, other, shndx = struct.unpack_from("<IIIBBH", b, s["off"] + k * s["entsize"])
                if size and info & 15 in (1, 2) and shndx < len(secs):
                    syms.append((size, cstr(names, nm), secs[shndx]["name"]))
    return secs, syms


def crate(name):
    m = re.match(r"_ZN(?:\d+\$LT\$)?(\d+)", name)
    if name.startswith("_ZN"):
        # legacy mangling: length-prefixed path components; the first that names a crate
        parts = []
        at = 3
        while at < len(name) and name[at].isdigit():
            j = at
            while name[j].isdigit():
                j += 1
            n = int(name[at:j])
            parts.append(name[j:j + n])
            at = j + n
        text = "::".join(parts)
        for c in ("klong_core", "klong_rt", "klong_probe", "libm", "compiler_builtins", "alloc", "core"):
            if re.search(r"(^|[^a-z_])" + c + r"(\.\.|::|\$)", text) or text.startswith(c):
                return c
        return "other"
    if re.match(r"__(aeabi|adddf|subdf|muldf|divdf|mulsf|divsf|addsf|subsf|fix|float|udiv|div|mod|umod|mul|ashl|lshr|ashr|extend|trunc|cmp|eq|ne|lt|le|gt|ge|unord|pow|clz|ctz)", name) or name in ("memcpy", "memmove", "memset", "memcmp", "bcmp", "compiler_builtins"):
        return "compiler_builtins"
    return "other"


def main(argv):
    path = argv[0]
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 12
    heap = int(argv[argv.index("--heap") + 1]) if "--heap" in argv else 0
    secs, syms = read(path)
    tot = {}
    for s in secs:
        if not s["flags"] & 2:  # SHF_ALLOC
            continue
        n = s["name"]
        key = ".text" if n.startswith(".text") else ".rodata" if n.startswith(".rodata") or n in (".ARM.exidx", ".ARM.extab", ".eh_frame") else ".bss" if s["type"] == 8 else ".data"
        tot[key] = tot.get(key, 0) + s["size"]
    tot[".bss"] = tot.get(".bss", 0) - heap
    print(f"{path}")
    print("  " + "  ".join(f"{k} {tot.get(k, 0)}" for k in (".text", ".rodata", ".data", ".bss")) + f"  flash {tot.get('.text', 0) + tot.get('.rodata', 0) + tot.get('.data', 0)}")
    if syms:
        by = {}
        for size, name, sec in syms:
            k = (crate(name), ".text" if sec.startswith(".text") else ".rodata" if sec.startswith(".rodata") else sec)
            by[k] = by.get(k, 0) + size
        print("  by crate (symbols with a size):")
        for (c, sec), n in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"    {n:8d}  {sec:8} {c}")
        print(f"  largest {top} symbols:")
        for size, name, sec in sorted(syms, reverse=True)[:top]:
            print(f"    {size:8d}  {sec:8} {name[:110]}")


if __name__ == "__main__":
    main(sys.argv[1:])
