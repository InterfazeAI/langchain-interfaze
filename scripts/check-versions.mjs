// Five files carry the version and two of them reach users as a User-Agent. A release
// cut from the wrong commit would otherwise publish the previous version, silently.
import { readFileSync } from "node:fs";

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");
const match = (path, re) => (read(path).match(re) ?? [])[1];

const versions = {
  "python/pyproject.toml": match("python/pyproject.toml", /^version = "(.+)"$/m),
  "python/interfaze_langchain/_version.py": match("python/interfaze_langchain/_version.py", /^__version__ = "(.+)"$/m),
  "js/package.json": JSON.parse(read("js/package.json")).version,
  "js/jsr.json": JSON.parse(read("js/jsr.json")).version,
  "js/src/version.ts": match("js/src/version.ts", /VERSION = "(.+)"/),
};

// With no tag argument the five just have to agree with each other, which is what the
// PR check wants; on a release they also have to agree with the tag.
const tag = process.argv[2]?.replace(/^v/, "") || versions["js/package.json"];
const wrong = Object.entries(versions).filter(([, version]) => version !== tag);

for (const [file, version] of wrong) console.error(`::error file=${file}::${version} does not match ${tag}`);
if (wrong.length) process.exit(1);
console.log(`all five versions are ${tag}`);
