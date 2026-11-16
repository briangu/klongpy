// Run a Klong program in the wasm32 size probe (klong-probe):
//   node scripts/probe.mjs FILE.wasm 'program' [STACK_BYTES]
// STACK_BYTES is the native stack budget of the evaluator (default 48 KiB; 0: no limit).
import { readFileSync } from "node:fs";
const [wasm, program, stack] = process.argv.slice(2);
let out = "";
let mem;
const { instance } = await WebAssembly.instantiate(readFileSync(wasm), {
  env: { host_write: (p, n) => { out += Buffer.from(mem.buffer, p, n).toString("utf8"); } },
});
mem = instance.exports.memory;
if (stack !== undefined) instance.exports.klong_set_stack(Number(stack));
const src = Buffer.from(program, "utf8");
const p = instance.exports.klong_alloc(src.length);
new Uint8Array(mem.buffer, p, src.length).set(src);
const code = instance.exports.klong_run(p, src.length);
console.log(out);
process.exit(code);
