// Run a Klong program in the wasm32 size probe (klong-probe): node scripts/probe.mjs FILE.wasm 'program'
import { readFileSync } from "node:fs";
const [wasm, program] = process.argv.slice(2);
let out = "";
let mem;
const { instance } = await WebAssembly.instantiate(readFileSync(wasm), {
  env: { host_write: (p, n) => { out += Buffer.from(mem.buffer, p, n).toString("utf8"); } },
});
mem = instance.exports.memory;
const src = Buffer.from(program, "utf8");
const p = instance.exports.klong_alloc(src.length);
new Uint8Array(mem.buffer, p, src.length).set(src);
const code = instance.exports.klong_run(p, src.length);
console.log(out);
process.exit(code);
