# JS `@interfaze/langchain` Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TypeScript LangChain integration (`@interfaze/langchain`, class `ChatInterfaze`) to this repo alongside the existing Python package, at behavioral parity with the Python `ChatInterfaze`.

**Architecture:** Reorganize the repo into `python/` + `js/`. The JS package subclasses `ChatOpenAICompletions` from `@langchain/openai` (the class that does the real Chat Completions work; the public `ChatOpenAI` is only a dispatcher) and overrides the constructor, `_generate`, `_streamResponseChunks`, and `_streamChatModelEvents` to inject `precontext`, rewrite video content blocks, surface Interfaze side-fields (`precontext`/`reasoning`/`vcache`), and strip `<think>`/`<precontext>` side-channel tags. It vendors `SideChannelFilter` + `stripSideChannels` locally (copied from the interfaze-js SDK, which is left untouched) and imports only `INTERFAZE_BASE_URL` / `INTERFAZE_MODEL` / `InterfazeError` from the published `interfaze` package.

**Tech Stack:** TypeScript (ES2022, NodeNext, strict), `@langchain/openai` 1.5.5, `@langchain/core` ^1.2.2, `interfaze` >=1.0.2, `zod` (optional peer), tsup (dual ESM/CJS), vitest, prettier, publint + are-the-types-wrong. Python side unchanged (uv, pytest, ruff, mypy).

## Global Constraints

- **npm name:** `@interfaze/langchain`. **JSR name:** `@interfaze-ai/langchain` (the base SDK's JSR scope is `@interfaze-ai`, its npm name is unscoped `interfaze`).
- **Pin `@langchain/openai` to exact `1.5.5`** (peerDep). The overrides ride on `@internal`/`@deprecated` surface that can shift on a minor bump.
- **Peer deps:** `@langchain/openai` `1.5.5`, `@langchain/core` `^1.2.2`, `interfaze` `>=1.0.2`; `zod` optional peer (`peerDependenciesMeta.zod.optional = true`).
- **Side-channel helpers are VENDORED**, not imported from `interfaze`. interfaze-js is mid-refactor on an uncommitted feature branch and must be left untouched (user decision, 2026-07-31). `SideChannelFilter` + `stripSideChannels` live in `js/src/side_channels.ts`. From `interfaze` we import ONLY `INTERFAZE_BASE_URL`, `INTERFAZE_MODEL`, `InterfazeError` (all present in the published 1.0.2).
- **Node engines:** `>=18`. Prettier `printWidth: 150`, `tabWidth: 2`, `singleQuote: false`, `trailingComma: "es5"`.
- **Never leak `__raw_response`** on output messages — strip it after reading side-fields.
- **All streaming surfaces must be filtered** — `.stream()` AND `.streamEvents()`. Filtering only `_streamResponseChunks` leaves `.streamEvents()` unfiltered.
- **Interfaze constants (import, don't hardcode):** `INTERFAZE_BASE_URL = "https://api.interfaze.ai/v1"`, `INTERFAZE_MODEL = "interfaze-beta"` from `interfaze`.
- **Side-field names:** `precontext`, `reasoning`, `vcache`. Apply each to BOTH `response_metadata` and `additional_kwargs`, only when non-null.
- **Out of scope:** no `tasks.*` / `guard` reimplementation. `withStructuredOutput`/`bindTools` inherit — no override.
- **Do NOT modify the interfaze-js repo** (`/Users/mylo/Work/interfaze-js`) in any task.

---

## File Structure

**Reorg (Task 1):**
- `python/` ← `git mv` of `langchain_interfaze/`, `tests/`, `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE` (keep a copy of LICENSE at root too).
- `README.md` (root, new) — intro + links to `python/README.md` and `js/README.md`.
- `.github/workflows/ci.yml`, `publish.yml` — split into python-scoped and js-scoped jobs.

**JS package (`js/`):**
- `js/package.json`, `js/tsconfig.json`, `js/tsup.config.ts`, `js/vitest.config.ts`, `js/.prettierrc`, `js/.prettierignore`, `js/jsr.json`, `js/LICENSE`, `js/README.md`
- `js/src/index.ts` — public re-exports.
- `js/src/side_channels.ts` — vendored `SideChannelFilter` + `stripSideChannels` (copied verbatim from interfaze-js's `src/stream.ts`; self-contained).
- `js/src/chat_models.ts` — `ChatInterfaze` class + private helpers (single file, mirroring Python's `chat_models.py`).
- `js/test/helpers.ts` — mock-`fetch` test harness.
- `js/test/side_channels.test.ts` — unit tests for the vendored helpers.
- `js/test/*.test.ts` — unit tests.
- `js/test/conformance.test.ts` — Runnable-surface conformance (invoke/batch/stream/pipe).

---

## Task 1: Repo reorganization (python/ + root README + workflows)

**Files:**
- Move: `langchain_interfaze/`, `tests/`, `pyproject.toml`, `uv.lock`, `README.md` → `python/`
- Create: `README.md` (root), `python/LICENSE` (copy)
- Modify: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`

**Interfaces:**
- Produces: a working `python/` package (`cd python && uv run pytest` passes) and a root README. No code symbols.

- [ ] **Step 1: Move Python content into `python/`**

```bash
mkdir python
git mv langchain_interfaze tests pyproject.toml uv.lock README.md python/
cp LICENSE python/LICENSE
```

`python/pyproject.toml` needs no path edits: `packages = ["langchain_interfaze"]`, `testpaths = ["tests/unit_tests"]`, `--cov=langchain_interfaze`, and `files = ["langchain_interfaze"]` are all relative to `python/`, which is the working directory when building/testing there.

- [ ] **Step 2: Verify the Python package still builds and tests from its new home**

Run: `cd python && uv sync --all-groups && uv run pytest tests/unit_tests/ -q`
Expected: all tests PASS (same suite as before the move).

- [ ] **Step 3: Write the root README**

Create `README.md` (root):

```markdown
# Interfaze LangChain SDKs

The official [LangChain](https://langchain.com) integrations for [Interfaze](https://interfaze.ai) — one class, `ChatInterfaze`, in both Python and TypeScript/JavaScript.

| Language | Package | Directory |
| --- | --- | --- |
| Python | [`langchain-interfaze`](https://pypi.org/project/langchain-interfaze/) (PyPI) | [`python/`](./python) |
| TypeScript / JavaScript | [`@interfaze/langchain`](https://www.npmjs.com/package/@interfaze/langchain) (npm) | [`js/`](./js) |

See [`python/README.md`](./python/README.md) and [`js/README.md`](./js/README.md) for install and usage.

Built on the core [Interfaze Python SDK](https://github.com/InterfazeAI/interfaze-python) and [Interfaze JS SDK](https://github.com/InterfazeAI/interfaze-js).

## License

MIT
```

- [ ] **Step 4: Repoint `ci.yml` — scope the Python job to `python/`, add a placeholder note for JS (JS jobs added in Task 12)**

Rewrite `.github/workflows/ci.yml` so the Python test job runs in `python/`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  python-test:
    name: python (py${{ matrix.python-version }})
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: python
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v9.0.0
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - name: Install
        run: uv sync --all-groups
      - name: Lint (ruff)
        run: uv run ruff check .
      - name: Type check (mypy)
        if: matrix.python-version == '3.12'
        run: uv run mypy
      - name: Unit tests
        run: uv run pytest tests/unit_tests/

  secret-scan:
    name: secret scan (gitleaks)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - name: Scan working tree for leaked tokens/secrets
        run: |
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.21.2/gitleaks_8.21.2_linux_x64.tar.gz" | tar -xz gitleaks
          ./gitleaks dir . --redact --no-banner
```

- [ ] **Step 5: Repoint `publish.yml` build/publish steps to `python/`**

In `.github/workflows/publish.yml`, add `working-directory: python` (via `defaults.run` on the `build` job) so `uv build` and `twine check dist/*` run in `python/`, and update the artifact `path:` to `python/dist/`. Leave the `testpypi`/`pypi` jobs' `download-artifact` `path: dist/` as-is (artifacts unpack relative to the runner default dir). Concretely, edit the `build` job:

```yaml
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: python
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v9.0.0
        with:
          python-version: "3.12"
      - name: Build sdist + wheel
        run: uv build
      - name: Verify metadata
        run: uvx twine check dist/*
      - uses: actions/upload-artifact@v7
        with:
          name: dist
          path: python/dist/
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move Python package into python/, add root README"
```

---

## Task 2: Vendor and test the side-channel helpers

> **Note:** this task assumes Task 3 (scaffold) has created `js/package.json` and installed dev deps, because it runs vitest. If executing strictly in order, do Task 3 first, then Task 2 — or run Task 2's `npm install`/`vitest` from a scaffolded `js/`. The controller may reorder Task 2 after Task 3.

**Files:**
- Create: `js/src/side_channels.ts`, `js/test/side_channels.test.ts`

**Interfaces:**
- Produces (from `../side_channels.js`): `class SideChannelFilter` with `feed(text: string): string` and `flush(): string`; `function stripSideChannels(content: string): { text: string; reasoning?: string; precontext?: Precontext[] }`; `type Precontext = Record<string, unknown>`. Consumed by `ChatInterfaze` in Tasks 6–7.
- Vendored verbatim from interfaze-js `src/stream.ts` (the SDK's own logic) so behavior stays identical. **Do not modify interfaze-js.**

- [ ] **Step 1: Write the failing tests**

`js/test/side_channels.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { SideChannelFilter, stripSideChannels } from "../src/side_channels.js";

describe("stripSideChannels", () => {
  it("pulls out reasoning and precontext, returns clean text", () => {
    const out = stripSideChannels(
      "<think>because</think><precontext>[{\"name\":\"ocr\"}]</precontext>The sky is blue."
    );
    expect(out.text).toBe("The sky is blue.");
    expect(out.reasoning).toBe("because");
    expect(out.precontext).toEqual([{ name: "ocr" }]);
  });

  it("leaves plain text untouched", () => {
    const out = stripSideChannels("just text");
    expect(out.text).toBe("just text");
    expect(out.reasoning).toBeUndefined();
    expect(out.precontext).toBeUndefined();
  });

  it("ignores a malformed precontext block", () => {
    const out = stripSideChannels("<precontext>not json</precontext>hi");
    expect(out.text).toBe("hi");
    expect(out.precontext).toBeUndefined();
  });
});

describe("SideChannelFilter", () => {
  it("strips a <think> block split across feed() calls", () => {
    const f = new SideChannelFilter();
    let out = "";
    for (const piece of ["<th", "ink>secret</think>The sky ", "is blue."]) out += f.feed(piece);
    out += f.flush();
    expect(out).toBe("The sky is blue.");
  });

  it("passes plain content straight through", () => {
    const f = new SideChannelFilter();
    expect(f.feed("Hello ") + f.feed("world") + f.flush()).toBe("Hello world");
  });

  it("drops an unterminated side-channel block on flush", () => {
    const f = new SideChannelFilter();
    expect(f.feed("<think>partial")).toBe("");
    expect(f.flush()).toBe("");
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd js && npx vitest run test/side_channels.test.ts`
Expected: FAIL (module `../src/side_channels.js` does not exist yet).

- [ ] **Step 3: Vendor the helpers**

Create `js/src/side_channels.ts` (copied verbatim from interfaze-js `src/stream.ts`, made self-contained with a local `Precontext` type):
```ts
// Vendored from the interfaze-js SDK (src/stream.ts) so the LangChain integration
// stays self-contained. Keep in sync if the SDK's side-channel logic changes.

export type Precontext = Record<string, unknown>;

const TAG_RE = (tag: string) => new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, "g");

/** Pull `<think>`/`<precontext>` blocks out of content; returns the rest as `text`. */
export function stripSideChannels(content: string): {
  text: string;
  reasoning?: string;
  precontext?: Precontext[];
} {
  let text = content;
  const thinks: string[] = [];
  text = text.replace(TAG_RE("think"), (_m, inner: string) => {
    thinks.push(inner.trim());
    return "";
  });
  const pre: Precontext[] = [];
  text = text.replace(TAG_RE("precontext"), (_m, inner: string) => {
    try {
      const parsed = JSON.parse(inner.trim());
      if (Array.isArray(parsed)) pre.push(...parsed);
      else pre.push(parsed);
    } catch {
      /* ignore malformed block */
    }
    return "";
  });
  const out: { text: string; reasoning?: string; precontext?: Precontext[] } = { text: text.trim() };
  if (thinks.length) out.reasoning = thinks.join("\n");
  if (pre.length) out.precontext = pre;
  return out;
}

const SIDE_OPEN = ["<think>", "<precontext>"] as const;
const SIDE_CLOSE: Record<string, string> = { "<think>": "</think>", "<precontext>": "</precontext>" };

function suffixPrefixLen(s: string, tag: string): number {
  for (let k = Math.min(s.length, tag.length - 1); k > 0; k--) {
    if (s.slice(s.length - k) === tag.slice(0, k)) return k;
  }
  return 0;
}

/** Strips inline `<think>`/`<precontext>` blocks from streamed content, chunk by chunk. */
export class SideChannelFilter {
  #buf = "";
  #close: string | undefined;

  feed(text: string): string {
    this.#buf += text;
    const out: string[] = [];
    while (this.#buf) {
      if (this.#close === undefined) {
        const lt = this.#buf.indexOf("<");
        if (lt === -1) {
          out.push(this.#buf);
          this.#buf = "";
          break;
        }
        if (lt > 0) {
          out.push(this.#buf.slice(0, lt));
          this.#buf = this.#buf.slice(lt);
        }
        const opened = SIDE_OPEN.find((t) => this.#buf.startsWith(t));
        if (opened) {
          this.#close = SIDE_CLOSE[opened];
          this.#buf = this.#buf.slice(opened.length);
          continue;
        }
        if (SIDE_OPEN.some((t) => t.startsWith(this.#buf))) break;
        out.push("<");
        this.#buf = this.#buf.slice(1);
      } else {
        const close = this.#close;
        const end = this.#buf.indexOf(close);
        if (end === -1) {
          const keep = suffixPrefixLen(this.#buf, close);
          this.#buf = keep ? this.#buf.slice(this.#buf.length - keep) : "";
          break;
        }
        this.#buf = this.#buf.slice(end + close.length);
        this.#close = undefined;
      }
    }
    return out.join("");
  }

  flush(): string {
    if (this.#close !== undefined) {
      this.#buf = "";
      return "";
    }
    const rest = this.#buf;
    this.#buf = "";
    return rest;
  }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd js && npx vitest run test/side_channels.test.ts`
Expected: all PASS.

- [ ] **Step 5: Typecheck**

Run: `cd js && npm run typecheck`
Expected: clean (the file is a verbatim copy of code that compiles under the same strict tsconfig in interfaze-js).

- [ ] **Step 6: Commit**

```bash
git add js/src/side_channels.ts js/test/side_channels.test.ts
git commit -m "feat(js): vendor side-channel helpers from the interfaze SDK"
```

---

## Task 3: Scaffold the JS package (`js/`)

**Files:**
- Create: `js/package.json`, `js/tsconfig.json`, `js/tsup.config.ts`, `js/vitest.config.ts`, `js/.prettierrc`, `js/.prettierignore`, `js/jsr.json`, `js/LICENSE`, `js/src/index.ts`, `js/src/chat_models.ts`

**Interfaces:**
- Produces: an installable, buildable empty package exporting a placeholder, plus a verified import path for `ChatOpenAICompletions`. Consumed by all later JS tasks.

- [ ] **Step 1: Create `js/package.json`**

```json
{
  "name": "@interfaze/langchain",
  "version": "1.0.0",
  "description": "Interfaze LangChain integration for TypeScript/JavaScript",
  "license": "MIT",
  "author": "InterfazeAI",
  "homepage": "https://interfaze.ai",
  "repository": {
    "type": "git",
    "url": "git+https://github.com/InterfazeAI/langchain-interfaze.git",
    "directory": "js"
  },
  "keywords": ["interfaze", "langchain", "openai", "ai", "llm"],
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": { "types": "./dist/index.d.ts", "default": "./dist/index.js" },
      "require": { "types": "./dist/index.d.cts", "default": "./dist/index.cjs" }
    },
    "./package.json": "./package.json"
  },
  "files": ["dist", "README.md", "LICENSE"],
  "sideEffects": false,
  "engines": { "node": ">=18" },
  "scripts": {
    "build": "tsup",
    "typecheck": "tsc --noEmit",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "check:pkg": "publint --strict && attw --pack",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "prepare": "tsup",
    "prepublishOnly": "npm run build"
  },
  "peerDependencies": {
    "@langchain/core": "^1.2.2",
    "@langchain/openai": "1.5.5",
    "interfaze": ">=1.0.2",
    "zod": "^3.23.0 || ^4.4.3"
  },
  "peerDependenciesMeta": {
    "zod": { "optional": true }
  },
  "devDependencies": {
    "@arethetypeswrong/cli": "0.18.5",
    "@langchain/core": "^1.2.2",
    "@langchain/openai": "1.5.5",
    "@types/node": "~22.20.1",
    "@vitest/coverage-v8": "^2.1.9",
    "interfaze": "^1.0.2",
    "prettier": "^3.9.6",
    "publint": "0.3.22",
    "tsup": "^8.5.1",
    "typescript": "~5.9.3",
    "vitest": "^2.1.9",
    "zod": "^4.4.3"
  }
}
```

- [ ] **Step 2: Create tooling configs**

`js/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022", "DOM"],
    "types": ["node"],
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": false,
    "declaration": true,
    "sourceMap": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "verbatimModuleSyntax": false,
    "outDir": "dist"
  },
  "include": ["src", "test"]
}
```

`js/tsup.config.ts`:
```ts
import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  target: "es2022",
  treeshake: true,
  splitting: false,
  external: ["@langchain/core", "@langchain/openai", "interfaze", "openai", "zod"],
});
```

`js/vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
    coverage: {
      provider: "v8",
      reporter: ["text", "text-summary", "lcov"],
      include: ["src/**/*.ts"],
      exclude: ["src/index.ts"],
      thresholds: { lines: 90, statements: 90, functions: 90, branches: 80 },
    },
  },
});
```

`js/.prettierrc`:
```json
{
  "printWidth": 150,
  "tabWidth": 2,
  "singleQuote": false,
  "bracketSameLine": false,
  "trailingComma": "es5"
}
```

`js/.prettierignore`:
```
dist
coverage
node_modules
```

`js/jsr.json`:
```json
{
  "name": "@interfaze-ai/langchain",
  "version": "1.0.0",
  "exports": "./src/index.ts",
  "publish": {
    "include": ["src", "README.md", "LICENSE", "jsr.json"]
  }
}
```

Copy the LICENSE: `cp LICENSE js/LICENSE` (run from repo root).

- [ ] **Step 3: Create placeholder source**

`js/src/chat_models.ts`:
```ts
export class ChatInterfaze {}
```

`js/src/index.ts`:
```ts
export { ChatInterfaze } from "./chat_models.js";
```

- [ ] **Step 4: Install deps**

`interfaze@1.0.2` is published, so a plain install resolves everything:

```bash
cd js && npm install
```

Expected: install succeeds; `js/node_modules/interfaze` is present and exports `INTERFAZE_BASE_URL`, `INTERFAZE_MODEL`, `InterfazeError`. (The side-channel helpers are vendored in `js/src/side_channels.ts` — Task 2 — not imported from `interfaze`.) If npm reports peer-dependency conflicts between `@langchain/openai@1.5.5` and the installed `@langchain/core`, reconcile by installing the `@langchain/core` version that satisfies the `@langchain/openai` peer range, not by forcing `--legacy-peer-deps`.

- [ ] **Step 5: Verify the `ChatOpenAICompletions` import path resolves**

Run:
```bash
cd js && node --input-type=module -e "import * as m from '@langchain/openai'; if (typeof m.ChatOpenAICompletions !== 'function') { console.error('exports:', Object.keys(m)); process.exit(1); } console.log('ChatOpenAICompletions ok');"
```
Expected: prints `ChatOpenAICompletions ok`. If it fails, inspect the printed `exports:` list — the class may be reachable via a subpath (e.g. `@langchain/openai/chat_models`); note the working import specifier and use it consistently in Task 4+.

- [ ] **Step 5b: Smoke-test the two load-bearing assumptions before building on them**

Every override in Tasks 4–8 rests on two facts from research that Task 3 is the first chance to confirm against the *installed* package: (a) `modelKwargs` lands as top-level request-body fields, and (b) `__includeRawResponse: true` stashes the raw response under `additional_kwargs.__raw_response`. Verify both directly on the base `ChatOpenAICompletions` (our subclass doesn't exist yet):

```bash
cd js && node --input-type=module -e '
import { ChatOpenAICompletions } from "@langchain/openai";
const calls = [];
const fetchImpl = async (input, init = {}) => {
  const raw = init.body ?? (input instanceof Request ? await input.clone().text() : undefined);
  calls.push(raw ? JSON.parse(raw) : undefined);
  return new Response(JSON.stringify({
    id: "x", object: "chat.completion", created: 1, model: "interfaze-beta",
    choices: [{ index: 0, message: { role: "assistant", content: "hi", refusal: null }, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }, vcache: false, reasoning: "r"
  }), { status: 200, headers: { "content-type": "application/json" } });
};
const m = new ChatOpenAICompletions({ apiKey: "t", model: "interfaze-beta", maxRetries: 0,
  configuration: { baseURL: "https://api.interfaze.ai/v1", fetch: fetchImpl },
  modelKwargs: { precontext: [{ name: "ocr" }] }, __includeRawResponse: true });
const res = await m.invoke("hi");
const bodyOk = JSON.stringify(calls.at(-1)?.precontext) === JSON.stringify([{ name: "ocr" }]);
const rawOk = !!res.additional_kwargs?.__raw_response && res.additional_kwargs.__raw_response.reasoning === "r";
console.log("modelKwargs->body:", bodyOk, "| __raw_response present:", rawOk);
if (!bodyOk || !rawOk) { console.log("body keys:", Object.keys(calls.at(-1) ?? {})); console.log("additional_kwargs:", res.additional_kwargs); process.exit(1); }
'
```
Expected: `modelKwargs->body: true | __raw_response present: true`. **If either is false, stop and reconcile before Task 4** — the printed `body keys` / `additional_kwargs` show where the data actually lives, and the constructor/`_generate` design must be adjusted to match (e.g. read the raw from a different key). Do not build Tasks 4–8 on an unconfirmed assumption.

- [ ] **Step 6: Verify build + typecheck of the scaffold**

Run: `cd js && npm run typecheck && npm run build`
Expected: typecheck clean; `dist/index.js`, `dist/index.cjs`, `dist/index.d.ts` emitted.

- [ ] **Step 7: Commit**

```bash
git add js/package.json js/package-lock.json js/tsconfig.json js/tsup.config.ts js/vitest.config.ts js/.prettierrc js/.prettierignore js/jsr.json js/LICENSE js/src
git commit -m "chore: scaffold @interfaze/langchain package"
```

---

## Task 4: Constructor — key resolution, defaults, precontext

**Files:**
- Modify: `js/src/chat_models.ts`, `js/src/index.ts`
- Create: `js/test/helpers.ts`, `js/test/constructor.test.ts`

**Interfaces:**
- Consumes: `INTERFAZE_BASE_URL`, `INTERFAZE_MODEL`, `InterfazeError` from `interfaze`; `ChatOpenAICompletions` from `@langchain/openai` (import specifier confirmed in Task 3 Step 5).
- Produces:
  - `interface ChatInterfazeFields` — the constructor options: `ChatOpenAIFields` (from `@langchain/openai`) plus `apiKey?: string`, `precontext?: Array<Record<string, unknown>>`.
  - `class ChatInterfaze extends ChatOpenAICompletions` with constructor `(fields?: ChatInterfazeFields)` that sets `this.lc_serializable = false` (overriding the base's `true`).
  - `js/test/helpers.ts` exporting `mockChat(responder, extraFields?)`, `completion(content?, extra?)`, `chunk(delta, finishReason?)`, `sseResponse(chunks)`, `jsonResponse(body)`, and constants `VIDEO_URL`, `CHAT_URL`.

- [ ] **Step 1: Write the test helper**

`js/test/helpers.ts`:
```ts
import { ChatInterfaze, type ChatInterfazeFields } from "../src/index.js";

export const CHAT_URL = "https://api.interfaze.ai/v1/chat/completions";
export const VIDEO_URL = "https://download.samplelib.com/mp4/sample-5s.mp4";

export interface CapturedRequest {
  url: string;
  body: Record<string, unknown> | undefined;
}

/** Build a ChatInterfaze whose OpenAI client uses a capturing mock `fetch`. */
export function mockChat(
  responder: (req: CapturedRequest) => Response,
  extraFields: Partial<ChatInterfazeFields> = {}
): { model: ChatInterfaze; calls: CapturedRequest[] } {
  const calls: CapturedRequest[] = [];
  const fetchImpl = async (input: unknown, init: RequestInit = {}): Promise<Response> => {
    const url = typeof input === "string" ? input : (input as Request).url;
    let raw: string | undefined = (init.body as string | undefined) ?? undefined;
    if (raw === undefined && input instanceof Request) raw = await input.clone().text();
    let body: Record<string, unknown> | undefined;
    try {
      body = raw ? (JSON.parse(raw) as Record<string, unknown>) : undefined;
    } catch {
      body = undefined;
    }
    const req: CapturedRequest = { url, body };
    calls.push(req);
    return responder(req);
  };
  const model = new ChatInterfaze({
    apiKey: "test-key",
    maxRetries: 0,
    configuration: { fetch: fetchImpl as unknown as never },
    ...extraFields,
  });
  return { model, calls };
}

export function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

export function sseResponse(chunks: unknown[]): Response {
  const body = `${chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("")}data: [DONE]\n\n`;
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

export function completion(content: unknown = "Hi!", extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: "req-test",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "interfaze-beta",
    choices: [{ index: 0, message: { role: "assistant", content, refusal: null }, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    vcache: false,
    ...extra,
  };
}

export function chunk(delta: Record<string, unknown>, finishReason: string | null = null): Record<string, unknown> {
  return {
    id: "req-test",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "interfaze-beta",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

export function lastBody(calls: CapturedRequest[]): Record<string, unknown> {
  const body = calls.at(-1)?.body;
  if (!body) throw new Error("no request captured");
  return body;
}
```

- [ ] **Step 2: Write the failing constructor tests**

`js/test/constructor.test.ts`:
```ts
import { afterEach, describe, expect, it } from "vitest";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";
import { ChatInterfaze } from "../src/index.js";
import { completion, jsonResponse, lastBody, mockChat } from "./helpers.js";

const KEY = "INTERFAZE_API_KEY";
afterEach(() => {
  delete process.env[KEY];
});

describe("ChatInterfaze constructor", () => {
  it("defaults baseURL and model to Interfaze", () => {
    const model = new ChatInterfaze({ apiKey: "t" });
    expect(model.model).toBe(INTERFAZE_MODEL);
    // clientConfig carries the resolved baseURL
    expect((model as unknown as { clientConfig: { baseURL?: string } }).clientConfig.baseURL).toBe(INTERFAZE_BASE_URL);
  });

  it("lets base url and model be overridden", () => {
    const model = new ChatInterfaze({ apiKey: "t", model: "other-model", configuration: { baseURL: "https://example.com/v1" } });
    expect(model.model).toBe("other-model");
    expect((model as unknown as { clientConfig: { baseURL?: string } }).clientConfig.baseURL).toBe("https://example.com/v1");
  });

  it("throws InterfazeError when no api key is present", () => {
    delete process.env[KEY];
    expect(() => new ChatInterfaze()).toThrow(InterfazeError);
    expect(() => new ChatInterfaze()).toThrow(/Missing API key/);
  });

  it("reads the api key from INTERFAZE_API_KEY", () => {
    process.env[KEY] = "env-key";
    expect(() => new ChatInterfaze()).not.toThrow();
  });

  it("is not lc-serializable (closes the real langchain-core serialization gate)", () => {
    const model = new ChatInterfaze({ apiKey: "t" });
    // langchain-core gates serialization on the `lc_serializable` field, not a method.
    // ChatOpenAI sets it true; ChatInterfaze must flip it false to match Python parity.
    expect(model.lc_serializable).toBe(false);
  });

  it("injects the precontext field into the request body", async () => {
    const pc = [{ name: "ocr", result: { extracted_text: "y" } }];
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")), { precontext: pc });
    await model.invoke("hi");
    expect(lastBody(calls).precontext).toEqual(pc);
  });

  it("omits precontext when not set", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")));
    await model.invoke("hi");
    expect("precontext" in lastBody(calls)).toBe(false);
  });
});
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd js && npx vitest run test/constructor.test.ts`
Expected: FAIL (placeholder `ChatInterfaze` has no constructor logic; `model.model` undefined, `lc_serializable` not overridden, precontext not injected).

- [ ] **Step 4: Implement the constructor**

Replace `js/src/chat_models.ts` with:
```ts
import { ChatOpenAICompletions, type ChatOpenAIFields } from "@langchain/openai";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";

export interface ChatInterfazeFields extends ChatOpenAIFields {
  /** Interfaze API key; falls back to `process.env.INTERFAZE_API_KEY`. */
  apiKey?: string;
  /** Precomputed tool output passed to Interfaze to skip its internal tool run. */
  precontext?: Array<Record<string, unknown>>;
}

export class ChatInterfaze extends ChatOpenAICompletions {
  constructor(fields: ChatInterfazeFields = {}) {
    const { apiKey, precontext, model, configuration, modelKwargs, ...rest } = fields;
    const key = apiKey ?? process.env.INTERFAZE_API_KEY;
    if (!key) {
      throw new InterfazeError(
        "Missing API key. Pass new ChatInterfaze({ apiKey: ... }) or set the INTERFAZE_API_KEY environment variable."
      );
    }
    super({
      ...rest,
      apiKey: key,
      model: model ?? INTERFAZE_MODEL,
      configuration: { baseURL: INTERFAZE_BASE_URL, ...configuration },
      modelKwargs: precontext !== undefined ? { ...modelKwargs, precontext } : modelKwargs,
    });
    // langchain-core gates serialization on the `lc_serializable` field (BaseChatOpenAI
    // sets it true). Flip it off so ChatInterfaze is not lc-serializable — Python parity
    // with `is_lc_serializable() -> False`. Set after super() so it wins over the base.
    this.lc_serializable = false;
  }
}
```

> `lc_serializable` is a plain boolean instance field on `BaseChatOpenAI` (not a getter). Assigning `this.lc_serializable = false` after `super()` is the robust override (a prototype getter would be shadowed by the base's own-property field, or throw on the base's assignment). Verify with the test that `new ChatInterfaze({ apiKey: "t" }).lc_serializable === false` AND construction does not throw.

> If Task 3 Step 5 found `ChatOpenAICompletions`/`ChatOpenAIFields` at a subpath, use that specifier here instead of `@langchain/openai`.

Also update `js/src/index.ts` to export the options type (`helpers.ts` imports it):
```ts
export { ChatInterfaze } from "./chat_models.js";
export type { ChatInterfazeFields } from "./chat_models.js";
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd js && npx vitest run test/constructor.test.ts`
Expected: all PASS. If the `clientConfig.baseURL` assertion path is wrong for this version, print the model with `console.log` once, find where the resolved base URL lives, and adjust the two assertions (implementation is correct if the request in the precontext test hits `api.interfaze.ai`).

- [ ] **Step 6: Commit**

```bash
git add js/src/chat_models.ts js/test/helpers.ts js/test/constructor.test.ts
git commit -m "feat(js): ChatInterfaze constructor (key, defaults, precontext)"
```

---

## Task 5: Video content-block rewrite

**Files:**
- Modify: `js/src/chat_models.ts`
- Create: `js/test/video.test.ts`

**Interfaces:**
- Consumes: `InterfazeError` from `interfaze`; `BaseMessage` from `@langchain/core/messages`.
- Produces: private method `rewriteVideoBlocks(messages: BaseMessage[]): BaseMessage[]` on `ChatInterfaze`, and an overridden `_generate` that applies it before delegating to `super._generate`. A `{type:"video", ...}` content part becomes `{type:"file", file:{...}}`.

- [ ] **Step 1: Write the failing video tests**

`js/test/video.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { HumanMessage } from "@langchain/core/messages";
import { InterfazeError } from "interfaze";
import { completion, jsonResponse, lastBody, mockChat, VIDEO_URL } from "./helpers.js";

function lastContent(calls: ReturnType<typeof mockChat>["calls"]): Array<Record<string, unknown>> {
  const messages = lastBody(calls).messages as Array<{ content: unknown }>;
  return messages.at(-1)!.content as Array<Record<string, unknown>>;
}

describe("video content blocks", () => {
  it("rewrites a url video block to a file part", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([
      new HumanMessage({ content: [{ type: "text", text: "what happens?" }, { type: "video", url: VIDEO_URL }] as never }),
    ]);
    expect(lastContent(calls)).toContainEqual({ type: "file", file: { file_data: VIDEO_URL } });
  });

  it("rewrites a base64 video block with mime type", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", base64: "AAAA", mime_type: "video/mp4" }] as never })]);
    const part = lastContent(calls)[0]!;
    expect(part).toEqual({ type: "file", file: { file_data: "data:video/mp4;base64,AAAA", format: "video/mp4" } });
  });

  it("rewrites a file_id video block", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", file_id: "file-123" }] as never })]);
    expect(lastContent(calls)[0]).toEqual({ type: "file", file: { file_id: "file-123" } });
  });

  it("forwards extras.filename", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([
      new HumanMessage({ content: [{ type: "video", url: VIDEO_URL, extras: { filename: "clip.mp4" } }] as never }),
    ]);
    const file = (lastContent(calls)[0]!.file as Record<string, unknown>);
    expect(file).toEqual({ file_data: VIDEO_URL, filename: "clip.mp4" });
  });

  it("throws when a video block has no source", async () => {
    const { model } = mockChat(() => jsonResponse(completion()));
    await expect(
      model.invoke([new HumanMessage({ content: [{ type: "video" }] as never })])
    ).rejects.toThrow(InterfazeError);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd js && npx vitest run test/video.test.ts`
Expected: FAIL (video blocks pass through unrewritten or error inside the OpenAI converter).

- [ ] **Step 3: Implement the rewrite + `_generate` override**

Add imports and methods to `js/src/chat_models.ts`. Add to the top imports:
```ts
import type { BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
```

Add these module-level helpers (above the class):
```ts
type VideoBlock = { type: "video"; url?: string; base64?: string; file_id?: string; mime_type?: string; extras?: { filename?: string } };

function convertVideoBlock(block: VideoBlock): Record<string, unknown> {
  let mime = block.mime_type;
  let file: Record<string, unknown>;
  // Key-existence checks (not truthiness) to match Python's `"url" in block`.
  if ("url" in block) {
    file = { file_data: block.url };
  } else if ("base64" in block) {
    mime = mime ?? "video/mp4";
    file = { file_data: `data:${mime};base64,${block.base64}` };
  } else if ("file_id" in block) {
    file = { file_id: block.file_id };
  } else {
    throw new InterfazeError("Video content block requires one of 'url', 'base64', or 'file_id'.");
  }
  // Python stamps `format` whenever mime is truthy (always, for base64). Match it.
  if (mime) file.format = mime;
  const filename = block.extras?.filename;
  if (filename) file.filename = filename;
  return { type: "file", file };
}

function rewriteContent(content: unknown): unknown {
  if (!Array.isArray(content)) return content;
  let changed = false;
  const out = content.map((block) => {
    if (block && typeof block === "object" && (block as { type?: string }).type === "video") {
      changed = true;
      return convertVideoBlock(block as VideoBlock);
    }
    return block;
  });
  return changed ? out : content;
}
```

Add these methods to the class:
```ts
  private rewriteVideoBlocks(messages: BaseMessage[]): BaseMessage[] {
    return messages.map((m) => {
      if (!Array.isArray(m.content)) return m;
      const rewritten = rewriteContent(m.content);
      if (rewritten === m.content) return m;
      const copy = Object.create(Object.getPrototypeOf(m));
      Object.assign(copy, m);
      copy.content = rewritten;
      return copy as BaseMessage;
    });
  }

  override async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    return super._generate(this.rewriteVideoBlocks(messages), options, runManager);
  }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd js && npx vitest run test/video.test.ts`
Expected: all PASS. If cloning via `Object.create` loses message behavior, fall back to the message's own copy helper: check whether `m` exposes a `.concat`/constructor-based clone in this `@langchain/core` version and use it; the required outcome is a new message whose `content` is the rewritten array without mutating the original.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `cd js && npx vitest run`
Expected: constructor + video tests PASS.

- [ ] **Step 6: Commit**

```bash
git add js/src/chat_models.ts js/test/video.test.ts
git commit -m "feat(js): rewrite video content blocks to file parts"
```

---

## Task 6: Non-streaming side-fields + tag stripping

**Files:**
- Modify: `js/src/chat_models.ts`
- Create: `js/test/side_fields.test.ts`

**Interfaces:**
- Consumes: `stripSideChannels` from `./side_channels.js` (vendored, Task 2); `AIMessage` from `@langchain/core/messages`; `__includeRawResponse` construction flag (sets `additional_kwargs.__raw_response` on generated messages).
- Produces: side-fields (`precontext`/`reasoning`/`vcache`) copied to `response_metadata` + `additional_kwargs`; `<think>`/`<precontext>` stripped from string content; `__raw_response` removed from output.

- [ ] **Step 1: Write the failing side-field tests**

`js/test/side_fields.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { AIMessage } from "@langchain/core/messages";
import { completion, jsonResponse, mockChat } from "./helpers.js";

const PC = [{ name: "ocr", result: { extracted_text: "x" } }];

describe("non-streaming side fields", () => {
  it("surfaces precontext/reasoning/vcache on both metadata maps", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Hello there", { precontext: PC, reasoning: "because reasons", vcache: true })));
    const res = (await model.invoke("hi")) as AIMessage;
    expect(res.response_metadata.precontext).toEqual(PC);
    expect(res.response_metadata.reasoning).toBe("because reasons");
    expect(res.response_metadata.vcache).toBe(true);
    expect(res.additional_kwargs.precontext).toEqual(PC);
    expect(res.additional_kwargs.reasoning).toBe("because reasons");
    expect(res.additional_kwargs.vcache).toBe(true);
  });

  it("leaves plain responses untouched and never leaks __raw_response", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Hi!")));
    const res = (await model.invoke("hi")) as AIMessage;
    expect(res.content).toBe("Hi!");
    expect("precontext" in res.response_metadata).toBe(false);
    expect("reasoning" in res.response_metadata).toBe(false);
    expect(res.response_metadata.vcache).toBe(false);
    expect("__raw_response" in res.additional_kwargs).toBe(false);
  });

  it("strips inline <think>/<precontext> tags from content", async () => {
    const content =
      "<think>Rayleigh scattering.</think>" +
      '<precontext>[{"name":"ocr","result":{"x":1}}]</precontext>' +
      "The sky is blue.";
    const { model } = mockChat(() => jsonResponse(completion(content)));
    const res = (await model.invoke("why is the sky blue?")) as AIMessage;
    expect(res.content).toBe("The sky is blue.");
    expect(res.response_metadata.reasoning).toBe("Rayleigh scattering.");
    expect(res.response_metadata.precontext).toEqual([{ name: "ocr", result: { x: 1 } }]);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd js && npx vitest run test/side_fields.test.ts`
Expected: FAIL (side-fields not surfaced, tags not stripped).

- [ ] **Step 3: Implement side-field surfacing in `_generate`**

Set `__includeRawResponse: true` in the `super()` call in the constructor — add it to the object passed to `super`:
```ts
    super({
      ...rest,
      apiKey: key,
      model: model ?? INTERFAZE_MODEL,
      configuration: { baseURL: INTERFAZE_BASE_URL, ...configuration },
      modelKwargs: precontext !== undefined ? { ...modelKwargs, precontext } : modelKwargs,
      __includeRawResponse: true,
    });
```
> `__includeRawResponse` is a valid `BaseChatOpenAI` field. If TS complains it's not on `ChatOpenAIFields`, cast the object: `super({ ... } as ChatOpenAIFields & { __includeRawResponse: boolean })`.

Add imports:
```ts
import { AIMessage } from "@langchain/core/messages";
import { stripSideChannels } from "./side_channels.js";
```

Add module-level helpers:
```ts
const SIDE_FIELDS = ["precontext", "reasoning", "vcache"] as const;

function applySideFields(message: AIMessage, raw: Record<string, unknown>): void {
  for (const key of SIDE_FIELDS) {
    const value = raw[key];
    if (value !== undefined && value !== null) {
      message.response_metadata[key] = value;
      message.additional_kwargs[key] = value as never;
    }
  }
}

function stripTags(message: AIMessage): void {
  if (typeof message.content !== "string") return;
  if (!message.content.includes("<think>") && !message.content.includes("<precontext>")) return;
  const { text, reasoning, precontext } = stripSideChannels(message.content);
  if (text !== message.content) message.content = text;
  if (reasoning && message.response_metadata.reasoning === undefined) {
    message.response_metadata.reasoning = reasoning;
    message.additional_kwargs.reasoning = reasoning as never;
  }
  if (precontext && message.response_metadata.precontext === undefined) {
    message.response_metadata.precontext = precontext;
    message.additional_kwargs.precontext = precontext as never;
  }
}
```

Rewrite `_generate` to post-process:
```ts
  override async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const result = await super._generate(this.rewriteVideoBlocks(messages), options, runManager);
    for (const generation of result.generations) {
      const message = generation.message;
      if (message instanceof AIMessage) {
        const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
        if (raw) applySideFields(message, raw);
        delete message.additional_kwargs.__raw_response;
        stripTags(message);
      }
    }
    return result;
  }
```

- [ ] **Step 4: Run to verify pass**

Run: `cd js && npx vitest run test/side_fields.test.ts`
Expected: all PASS. If `__raw_response` isn't present on `additional_kwargs`, `console.log(message.additional_kwargs)` once to find where `__includeRawResponse` stashes it in this version, and read from there (the intended source is the full raw `chat.completion` body carrying the top-level `precontext`/`reasoning`/`vcache`).

- [ ] **Step 5: Run the full suite**

Run: `cd js && npx vitest run`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add js/src/chat_models.ts js/test/side_fields.test.ts
git commit -m "feat(js): surface Interfaze side-fields and strip side-channel tags"
```

---

## Task 7: Streaming — side-channel filter + final side-field chunk

**Files:**
- Modify: `js/src/chat_models.ts`
- Create: `js/test/stream.test.ts`

**Interfaces:**
- Consumes: `SideChannelFilter`, `stripSideChannels` from `./side_channels.js` (vendored, Task 2); `AIMessageChunk` from `@langchain/core/messages`; `ChatGenerationChunk` from `@langchain/core/outputs`; the module-level `applySideFields` helper defined in Task 6.
- Produces: overridden `_streamResponseChunks(messages, options, runManager?): AsyncGenerator<ChatGenerationChunk>` that rewrites video blocks, attaches per-chunk top-level side-fields (Python parity with `_convert_chunk_to_generation_chunk`), strips `__raw_response` from every chunk, filters `<think>`/`<precontext>` from streamed text across chunk boundaries, and emits a trailing `ChatGenerationChunk` carrying the flushed tail + `reasoning`/`precontext`.

- [ ] **Step 1: Write the failing streaming tests**

`js/test/stream.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { chunk, mockChat, sseResponse } from "./helpers.js";

async function collect(model: { stream: (i: string) => Promise<AsyncIterable<{ content: unknown; additional_kwargs: Record<string, unknown> }>> }) {
  const out: Array<{ content: unknown; additional_kwargs: Record<string, unknown> }> = [];
  for await (const c of await model.stream("x")) out.push(c);
  return out;
}

describe("streaming side-channel filter", () => {
  it("strips inline precontext and carries it on a chunk", async () => {
    const chunks = [
      chunk({ content: '<precontext>[{"name":"ocr","result":{"extracted_text":"x"}}]</precontext>' }),
      chunk({ content: "Total " }),
      chunk({ content: "is $12.34" }),
      chunk({}, "stop"),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(text).not.toContain("<precontext>");
    expect(text).toBe("Total is $12.34");
    const withPc = got.filter((c) => c.additional_kwargs.precontext);
    expect(withPc.length).toBeGreaterThan(0);
    expect((withPc[0]!.additional_kwargs.precontext as Array<{ name: string }>)[0]!.name).toBe("ocr");
  });

  it("recovers <think> reasoning split across chunk boundaries", async () => {
    const chunks = [
      chunk({ content: "<th" }),
      chunk({ content: "ink>Rayleigh scat" }),
      chunk({ content: "tering.</think>The sky " }),
      chunk({ content: "is blue." }),
      chunk({}, "stop"),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(text).not.toContain("<think>");
    expect(text).toBe("The sky is blue.");
    const reasoning = got.filter((c) => c.additional_kwargs.reasoning);
    expect(reasoning[0]!.additional_kwargs.reasoning).toBe("Rayleigh scattering.");
  });

  it("emits no side-channel chunk for plain content", async () => {
    const chunks = [chunk({ content: "Hello " }), chunk({ content: "world" }), chunk({}, "stop")];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(text).toBe("Hello world");
    expect(got.some((c) => c.additional_kwargs.precontext || c.additional_kwargs.reasoning)).toBe(false);
    // never leak the raw response on streamed chunks
    expect(got.some((c) => "__raw_response" in c.additional_kwargs)).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd js && npx vitest run test/stream.test.ts`
Expected: FAIL (tags not stripped in stream; no side-field chunk).

- [ ] **Step 3: Implement `_streamResponseChunks` override**

Add imports:
```ts
import { AIMessageChunk } from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import { SideChannelFilter } from "./side_channels.js";
```

Add the override to the class:
```ts
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const filter = new SideChannelFilter();
    const rawParts: string[] = [];
    for await (const gen of super._streamResponseChunks(this.rewriteVideoBlocks(messages), options, runManager)) {
      const message = gen.message;
      if (message instanceof AIMessageChunk) {
        // Python parity: read top-level side-fields off each chunk's raw, then
        // strip __raw_response so it never leaks (see Task 6's no-leak rule).
        const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
        if (raw) applySideFields(message, raw);
        delete message.additional_kwargs.__raw_response;
        if (typeof message.content === "string" && message.content) {
          rawParts.push(message.content);
          message.content = filter.feed(message.content);
        }
      }
      yield gen;
    }
    const tail = filter.flush();
    const { reasoning, precontext } = stripSideChannels(rawParts.join(""));
    if (!tail && !reasoning && !precontext) return;
    const finalMessage = new AIMessageChunk({ content: tail });
    if (reasoning) {
      finalMessage.response_metadata.reasoning = reasoning;
      finalMessage.additional_kwargs.reasoning = reasoning;
    }
    if (precontext) {
      finalMessage.response_metadata.precontext = precontext;
      finalMessage.additional_kwargs.precontext = precontext as never;
    }
    yield new ChatGenerationChunk({ message: finalMessage, text: tail });
  }
```

- [ ] **Step 4: Run to verify pass**

Run: `cd js && npx vitest run test/stream.test.ts`
Expected: all PASS. If `ChatGenerationChunk` requires additional fields (e.g. `generationInfo`) in this version, add them per the type error; the required behavior is the assertions above.

> **Executor note (verify, don't assume):** we filter `message.content`, but `ChatGenerationChunk` also carries a `text` field. Confirm downstream consumers (output parsers, `.streamEvents()`) read `content`, not `text`. If any read `text`, set it from the filtered content too (`gen.text = message.content` when content is a string) so unfiltered text can't leak through that field.

- [ ] **Step 5: Run the full suite**

Run: `cd js && npx vitest run`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add js/src/chat_models.ts js/test/stream.test.ts
git commit -m "feat(js): filter side channels while streaming"
```

---

## Task 8: `.streamEvents()` must be filtered too

**Files:**
- Modify: `js/src/chat_models.ts`
- Create: `js/test/stream_events.test.ts`

**Interfaces:**
- Consumes: the overridden `_streamResponseChunks` from Task 7.
- Produces: `.streamEvents()` yields the same filtered content as `.stream()` — `ChatOpenAICompletions` ships a native `_streamChatModelEvents` that bypasses `_streamResponseChunks`, so it must be neutralized.

- [ ] **Step 1: Write the failing test**

`js/test/stream_events.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { chunk, mockChat, sseResponse } from "./helpers.js";

describe(".streamEvents() filtering", () => {
  it("strips side-channel tags from streamed events", async () => {
    const chunks = [
      chunk({ content: "<th" }),
      chunk({ content: "ink>secret</think>The sky " }),
      chunk({ content: "is blue." }),
      chunk({}, "stop"),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    let text = "";
    for await (const ev of model.streamEvents("x", { version: "v2" })) {
      if (ev.event === "on_chat_model_stream") {
        const c = (ev.data as { chunk?: { content?: unknown } }).chunk?.content;
        if (typeof c === "string") text += c;
      }
    }
    expect(text).not.toContain("<think>");
    expect(text).not.toContain("secret");
    expect(text).toBe("The sky is blue.");
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd js && npx vitest run test/stream_events.test.ts`
Expected: FAIL — the streamed event content still contains `<think>secret</think>` because the native `_streamChatModelEvents` bypasses our filter.

- [ ] **Step 3: Neutralize the native events path**

First inspect the installed implementation to choose the minimal override:
```bash
cd js && node -e "const p=require.resolve('@langchain/openai'); console.log(p);"
grep -rn "_streamChatModelEvents" js/node_modules/@langchain/openai/dist || true
```

Preferred implementation — override `_streamChatModelEvents` to fall back to core's generic bridge (which drives `.streamEvents()` off our filtered `_streamResponseChunks`) instead of the OpenAI-native fast path. Add to the class:
```ts
  // ChatOpenAICompletions overrides _streamChatModelEvents with a native path that
  // bypasses _streamResponseChunks (and therefore our side-channel filter). Restore
  // core's default so .streamEvents() streams the filtered chunks.
  override async *_streamChatModelEvents(
    ...args: Parameters<ChatOpenAICompletions["_streamChatModelEvents"]>
  ): ReturnType<ChatOpenAICompletions["_streamChatModelEvents"]> {
    const BaseChatModel = Object.getPrototypeOf(Object.getPrototypeOf(ChatOpenAICompletions.prototype));
    yield* BaseChatModel._streamChatModelEvents.apply(this, args);
  }
```

If the grandparent lookup resists on the **first** attempt (core doesn't expose `_streamChatModelEvents`, or the prototype chain differs), switch to the fallback immediately — do not spend turns tuning the prototype walk. Fallback: reimplement the event stream directly from our filtered chunks. Ask the failing test to confirm the event shape (`ev.event`, `ev.data.chunk`), then emit matching events from `this._streamResponseChunks(...)`. The required outcome is only the test's assertion: `.streamEvents()` content equals the filtered `.stream()` content.

**A passing test with no override at all is a valid outcome.** If Step 2 shows the test already passes — because the installed version has no native bypass, or already routes `.streamEvents()` through `_streamResponseChunks` — do not add an override to force one into existence. Delete the empty test scaffold's TODO, keep the test as a regression guard, and move on. The test tells the truth about the installed version.

- [ ] **Step 4: Run to verify pass**

Run: `cd js && npx vitest run test/stream_events.test.ts`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `cd js && npx vitest run`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add js/src/chat_models.ts js/test/stream_events.test.ts
git commit -m "feat(js): filter side channels through .streamEvents() too"
```

---

## Task 9: `withStructuredOutput` + `bindTools` smoke tests (inherited)

**Files:**
- Create: `js/test/structured_and_tools.test.ts`

**Interfaces:**
- Consumes: inherited `withStructuredOutput` and `bindTools` from `BaseChatOpenAI`. No production code changes — these are guardrail tests proving inheritance survives the subclass.

- [ ] **Step 1: Write the tests**

`js/test/structured_and_tools.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { completion, jsonResponse, mockChat } from "./helpers.js";

describe("inherited capabilities", () => {
  it("withStructuredOutput parses a JSON response", async () => {
    const body = completion(JSON.stringify({ merchant: "Walmart", total: 144.02 }));
    const { model } = mockChat(() => jsonResponse(body));
    const structured = model.withStructuredOutput(z.object({ merchant: z.string(), total: z.number() }), { method: "jsonMode" });
    const out = (await structured.invoke("extract")) as { merchant: string; total: number };
    expect(out.merchant).toBe("Walmart");
    expect(out.total).toBe(144.02);
  });

  it("bindTools surfaces tool_calls", async () => {
    const toolCall = {
      id: "call_1",
      type: "function",
      function: { name: "get_weather", arguments: JSON.stringify({ city: "Tokyo" }) },
    };
    const body = completion(null, { choices: [{ index: 0, message: { role: "assistant", content: null, tool_calls: [toolCall] }, finish_reason: "tool_calls" }] });
    const { model } = mockChat(() => jsonResponse(body));
    const bound = model.bindTools([
      { type: "function", function: { name: "get_weather", description: "Weather for a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } },
    ]);
    const res = await bound.invoke("weather in Tokyo?");
    expect(res.tool_calls?.[0]?.name).toBe("get_weather");
    expect(res.tool_calls?.[0]?.args).toEqual({ city: "Tokyo" });
  });
});
```

- [ ] **Step 2: Run the tests**

Run: `cd js && npx vitest run test/structured_and_tools.test.ts`
Expected: PASS. If `jsonMode` isn't the right method for a mocked endpoint, switch to `{ method: "functionCalling" }` and mock a tool-call response for the structured test; the point is that the inherited method runs through the subclass. If a `zod` version mismatch appears, align the `zod` devDependency with the `@langchain/openai` peer range.

> **Executor note (accepted parity gap):** this test mocks clean JSON. Real Interfaze may return `json_object` content wrapped in a ```` ```json ```` fence; the SDK exports `stripJsonFence` for exactly that. We do **not** strip the fence here — this matches the Python package (which also doesn't), so it's deliberate parity, not an oversight. If a real fence issue surfaces later, that's a shared cross-language decision, not a JS-only bug.

- [ ] **Step 3: Commit**

```bash
git add js/test/structured_and_tools.test.ts
git commit -m "test(js): structured output and tool calling smoke tests"
```

---

## Task 10: Runnable-surface conformance suite

> **Replan note (2026-07-31):** the original plan used `@langchain/standard-tests` (the JS analog of Python's `langchain-tests`). That package is **not published to npm** (verified `E404`; it's a monorepo-internal workspace package). There is no public external equivalent, so this task instead exercises the standard LangChain Runnable surface directly through the mock, proving `ChatInterfaze` behaves as a conforming chat model. This is the pragmatic parity substitute forced by ecosystem reality.

**Files:**
- Create: `js/test/conformance.test.ts`

**Interfaces:**
- Consumes: `ChatInterfaze` and the Task 4 `test/helpers.ts`. Drives the public Runnable API (`invoke`, `batch`, `stream`, `.pipe()` in an LCEL chain, and serialization guards) — the surface Python's `ChatModelUnitTests` covers.

- [ ] **Step 1: Write the conformance tests**

`js/test/conformance.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatInterfaze } from "../src/index.js";
import { completion, jsonResponse, mockChat, sseResponse, chunk } from "./helpers.js";

describe("ChatInterfaze conforms to the standard chat-model surface", () => {
  it("invoke returns an AIMessage with string content", async () => {
    const { model } = mockChat(() => jsonResponse(completion("hello")));
    const res = await model.invoke("hi");
    expect(res.content).toBe("hello");
    expect(res.getType()).toBe("ai");
  });

  it("batch fans out over multiple inputs", async () => {
    const { model } = mockChat(() => jsonResponse(completion("ok")));
    const out = await model.batch(["a", "b", "c"]);
    expect(out).toHaveLength(3);
    expect(out.every((m) => m.content === "ok")).toBe(true);
  });

  it("stream yields message chunks", async () => {
    const { model } = mockChat(() => sseResponse([chunk({ content: "Hel" }), chunk({ content: "lo" }), chunk({}, "stop")]));
    let text = "";
    for await (const c of await model.stream("hi")) text += typeof c.content === "string" ? c.content : "";
    expect(text).toBe("Hello");
  });

  it("composes in an LCEL chain via .pipe()", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Bonjour")));
    const chain = ChatPromptTemplate.fromTemplate("Translate to {lang}: {text}").pipe(model).pipe(new StringOutputParser());
    const out = await chain.invoke({ lang: "French", text: "Hello" });
    expect(out).toBe("Bonjour");
  });

  it("is not lc-serializable and exposes the standard identifiers", () => {
    const model = new ChatInterfaze({ apiKey: "t" });
    expect(model.lc_serializable).toBe(false);
    expect(model._llmType()).toBeTypeOf("string");
    expect(model._modelType()).toBeTypeOf("string");
  });
});
```

- [ ] **Step 2: Run the suite**

Run: `cd js && npx vitest run test/conformance.test.ts`
Expected: PASS. If `_llmType`/`_modelType`/`getType` have different names in the installed `@langchain/core`, adjust to the real method names (check `node_modules/@langchain/core/dist` types); the required outcome is that the public Runnable surface (invoke/batch/stream/pipe) works through the subclass.

- [ ] **Step 3: Run the full suite + coverage**

Run: `cd js && npm run test:coverage`
Expected: all PASS; coverage meets thresholds (lines/statements/functions ≥ 90, branches ≥ 80). If a branch is uncovered, add a targeted test rather than lowering the threshold.

- [ ] **Step 4: Commit**

```bash
git add js/test/conformance.test.ts
git commit -m "test(js): Runnable-surface conformance suite"
```

---

## Task 11: `js/README.md`

**Files:**
- Create: `js/README.md`

**Interfaces:**
- Produces: user-facing docs. TS port of `python/README.md`, section-for-section.

- [ ] **Step 1: Write `js/README.md`**

Port `python/README.md` to TypeScript idiom. Required sections (mirror the Python README order): title + badge links; Install (`npm install @interfaze/langchain`, note peers `@langchain/openai @langchain/core interfaze`); Setup (`new ChatInterfaze({ apiKey })` or `INTERFAZE_API_KEY`); Your first request (structured output from an image using `withStructuredOutput` + `z.object`, reading `precontext` off `response_metadata`); Precontext (`response_metadata.precontext / reasoning / vcache`); Chat (`invoke` with `SystemMessage`/`HumanMessage`); Streaming (`for await (const chunk of await llm.stream(...))`); Structured output (`withStructuredOutput`, `includeRaw`); Tools (`bindTools`, `tool_calls`); Reasoning (`reasoningEffort`); Multimodal inputs (file parts + `{ type: "video", ... }`); Async and batch (`invoke`/`stream`/`batch` are async; note no separate sync API); Chains (LCEL via `.pipe()`); Feeding precontext (`new ChatInterfaze({ precontext: [...] })`); Tasks / guardrails (punt to the core `interfaze` client); Errors (`InterfazeError` for client-side, `APIError` subclasses from `interfaze`); Capabilities table; License. Use fenced ` ```ts ` blocks. Keep prose lifted from the Python README where it applies verbatim.

Every code block must use real API surface built in Tasks 4–10 (`ChatInterfaze`, `withStructuredOutput`, `bindTools`, `stream`, `streamEvents`, the `{ type: "video" }` block, the `precontext` constructor field).

- [ ] **Step 2: Verify code blocks type-check**

Extract each ` ```ts ` block into a scratch `js/tmp-readme-check.ts`, prepend the imports, and run `cd js && npx tsc --noEmit tmp-readme-check.ts` (or paste into an existing typechecked scratch). Fix any signature drift. Delete the scratch file afterward.

- [ ] **Step 3: Commit**

```bash
git add js/README.md
git commit -m "docs(js): README for @interfaze/langchain"
```

---

## Task 12: JS CI + publish workflow jobs

**Files:**
- Modify: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`

**Interfaces:**
- Produces: CI jobs that build/typecheck/test the JS package on Node 20/22/24, and publish jobs for npm + JSR. (No interfaze-js publish dependency — `interfaze@1.0.2` is already on the registry and the side-channel helpers are vendored.)

- [ ] **Step 1: Add JS jobs to `ci.yml`**

Append to `.github/workflows/ci.yml` `jobs:` (mirroring interfaze-js's CI), all scoped to `js/`:
```yaml
  js-static:
    name: js static (format · types · build · package)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: js
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v7
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: js/package-lock.json
      - run: npm ci
      - run: npm run format:check
      - run: npm run typecheck
      - run: npm run build
      - run: npm run check:pkg

  js-test:
    name: js test (node ${{ matrix.node }})
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: js
    strategy:
      fail-fast: false
      matrix:
        node: [20, 22, 24]
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v7
        with:
          node-version: ${{ matrix.node }}
          cache: npm
          cache-dependency-path: js/package-lock.json
      - run: npm ci
      - run: npm run test:coverage
```

- [ ] **Step 2: Add JS publish jobs to `publish.yml`**

Append npm + JSR jobs (mirroring interfaze-js's publish workflow), scoped to `js/`:
```yaml
  npm-publish:
    name: Publish to npm
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    defaults:
      run:
        working-directory: js
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v7
        with:
          node-version: "24.x"
          registry-url: "https://registry.npmjs.org"
      - run: npm ci
      - run: npm run build
      - run: npm run check:pkg
      - run: npm publish --provenance --access public

  jsr-publish:
    name: Publish to JSR
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    defaults:
      run:
        working-directory: js
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v7
        with:
          node-version: "22.x"
      - run: npm ci
      - run: npx jsr publish --allow-slow-types
```

> Note the existing Python publish jobs trigger on `release: published`. If a single GitHub release should not publish both ecosystems at once, gate these JS jobs (and the Python ones) on a tag prefix or `workflow_dispatch` input in a follow-up; for now they share the release trigger. Flag this to the maintainer.

- [ ] **Step 3: Lint the workflow YAML locally**

Run: `cd js && npx prettier --check ../.github/workflows/ci.yml ../.github/workflows/publish.yml` (or a YAML linter if available). Fix formatting.
Expected: valid YAML, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/publish.yml
git commit -m "ci: build, test, and publish the JS package"
```

---

## Self-Review

**Spec coverage:**
- Repo reorg (python/ + js/, root README, workflow repoint) → Task 1, Task 12. ✓
- JS tooling mirrors interfaze-js (tsup dual, vitest, prettier 150, peer deps, pinned openai) → Task 3, Global Constraints. ✓
- Constructor (key/env, defaults, precontext→modelKwargs, lc_serializable=false) → Task 4. ✓
- Video block rewrite (url/base64/file_id/filename/error) → Task 5. ✓
- Side-fields + tag stripping + no `__raw_response` leak → Task 6. ✓
- Streaming filter + final chunk → Task 7. ✓
- `.streamEvents()` bypass fix → Task 8. ✓
- withStructuredOutput/bindTools inherit → Task 9. ✓
- Runnable-surface conformance (standard-tests unavailable on npm) → Task 10. ✓
- js/README → Task 11. ✓
- vendored side-channel helpers (interfaze-js left untouched) → Task 2. ✓
- CI/publish (npm + JSR) → Task 12. ✓
- Out of scope (no tasks/guard) → honored (no task adds them). ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Each code step carries real code. The two genuinely version-dependent spots (`ChatOpenAICompletions` import path in Task 3.5; `_streamChatModelEvents` neutralization in Task 8.3) carry a concrete primary implementation plus a test-driven fallback, not a placeholder.

**Type consistency:** `rewriteVideoBlocks(messages: BaseMessage[]): BaseMessage[]` defined in Task 5, reused verbatim in Tasks 6–7. `applySideFields`/`stripTags`/`convertVideoBlock`/`rewriteContent`/`SIDE_FIELDS` defined once (Tasks 5–6), consumed consistently. `ChatInterfazeFields` (Task 4) extended in place. Side-field name list (`precontext`/`reasoning`/`vcache`) consistent across Tasks 6–7 and Global Constraints. Helper signatures in `test/helpers.ts` (Task 4) match all consuming tests.
```
