# Design: JS/TS `@interfaze/langchain` alongside the Python package

**Date:** 2026-07-31
**Status:** Approved (design), pending spec review
**Repo:** `langchain-interfaze`

## Problem

This repo currently holds only the **Python** LangChain integration for Interfaze
(`langchain_interfaze/`, published to PyPI as `langchain-interfaze`). The
JavaScript/TypeScript equivalent is missing. Both the Python and the JS LangChain
integrations should live in this single repo, with the JS build published to npm
(and JSR) as `@interfaze/langchain`.

The JS integration must reach behavioral parity with the Python `ChatInterfaze`,
built on the `interfaze-js` SDK the same way the Python package builds on the
`interfaze-py` SDK.

## Context / prior art

- **Python integration** (`langchain_interfaze/chat_models.py`): `ChatInterfaze`
  subclasses `ChatOpenAI` from `langchain-openai` and overrides
  `__init__`, `_get_request_payload`, `_create_chat_result`,
  `_convert_chunk_to_generation_chunk`, `_stream`, `_astream`. It consumes
  `INTERFAZE_BASE_URL`, `INTERFAZE_MODEL`, `InterfazeError`, `SideChannelFilter`,
  `strip_side_channels` from the `interfaze` (interfaze-py) SDK.
- **interfaze-js SDK** (`/Users/mylo/Work/interfaze-js`): the JS SDK. Has the
  identical helpers — `SideChannelFilter` (private class) and `stripSideChannels`
  (exported from `stream.ts` but not re-exported from `index.ts`) — plus
  `INTERFAZE_BASE_URL`, `INTERFAZE_MODEL`, `InterfazeError`. Tooling: tsup dual
  ESM/CJS, vitest + JSON fixtures, prettier (printWidth 150), dual npm+JSR
  publish, CI matrix on Node 20/22/24.
- **`@langchain/openai`** (npm): latest **1.5.5**, peer `@langchain/core ^1.2.2`.
  The class layout was refactored vs. Python: the public `ChatOpenAI` (in
  `chat_models/index.ts`) is a **dispatcher** holding `this.completions` and
  `this.responses`; the actual Chat Completions work lives on
  **`ChatOpenAICompletions`** (`chat_models/completions.ts`, tagged `@internal`
  but exported). Interfaze is an OpenAI-compatible **Chat Completions** endpoint,
  so the Responses path is dead weight.

## Decisions (locked with the user)

1. **Repo layout:** symmetric `python/` + `js/` split.
2. **npm name + registries:** `@interfaze/langchain`, published to **both npm and JSR**
   (mirror interfaze-js). Note: the `@interfaze` npm scope is not yet claimed —
   the owner must register/own it before publishing.
3. **Side-channel helpers:** export `SideChannelFilter` + `stripSideChannels` from
   interfaze-js and import them here (full parity with how the Python package
   consumes interfaze-py) — rather than vendoring a local copy.

## Repo reorganization

```
langchain-interfaze/
  python/                      # git mv of all current Python content
    langchain_interfaze/
    tests/
    pyproject.toml
    uv.lock
    README.md                  # current root README moves here
  js/                          # NEW TypeScript package
    src/
      index.ts
      chat_models.ts
    test/
      *.test.ts
      fixtures/*.json
    package.json               # @interfaze/langchain
    tsconfig.json
    tsup.config.ts
    vitest.config.ts
    jsr.json
    .prettierrc
    .prettierignore
    README.md
  .github/workflows/
    ci.yml                     # python-* and js-* jobs (working-directory per job)
    publish.yml                # python (PyPI) + js (npm + JSR) publish jobs
  README.md                    # NEW root: short intro + links to python/ and js/
  LICENSE
```

- `python/pyproject.toml` gets its paths repointed relative to `python/`:
  `[tool.hatch.build.targets.wheel] packages`, `[tool.pytest.ini_options] testpaths`,
  `--cov=langchain_interfaze`, `[tool.mypy] files`. Because the package still builds
  from inside `python/`, most values are unchanged; only the working directory shifts.
- Both workflows set `working-directory` (or per-job `defaults.run.working-directory`)
  so Python jobs run in `python/` and JS jobs run in `js/`.
- Nothing is published to PyPI or npm yet, so this reorg breaks nothing downstream.

## JS package (`js/`)

Tooling mirrors interfaze-js exactly:

- **package.json**: `name: "@interfaze/langchain"`, `type: "module"`,
  `main`/`module`/`types`/`exports` dual ESM+CJS (`dist/index.js`, `dist/index.cjs`,
  `dist/index.d.ts` / `.d.cts`), `files: ["dist", "README.md", "LICENSE"]`,
  `sideEffects: false`, `engines.node >= 18`.
  Scripts: `build` (tsup), `typecheck`, `format`/`format:check`, `check:pkg`
  (publint + attw), `test`, `test:coverage`, `prepare`/`prepublishOnly`.
- **dependencies:** none bundled.
- **peerDependencies:**
  - `@langchain/openai` — **pinned exact** (e.g. `1.5.5`) because the port leans on
    `@internal`/`@deprecated` surface that can shift on a minor bump.
  - `@langchain/core` — `^1.2.2` (the `@langchain/openai` 1.5.x peer).
  - `interfaze` — `>=1.0.3` (the version that adds the helper exports).
  - `zod` — optional peer (`peerDependenciesMeta.zod.optional = true`), used by
    `withStructuredOutput` with a Zod schema.
- **devDependencies:** `@langchain/openai`, `@langchain/core`, `@langchain/standard-tests`,
  `interfaze`, `zod`, `tsup`, `tsx`, `typescript`, `vitest`, `@vitest/coverage-v8`,
  `prettier`, `publint`, `@arethetypeswrong/cli`, `@types/node`.
- **tsconfig.json / tsup.config.ts / vitest.config.ts / .prettierrc**: copied from
  interfaze-js (ES2022, NodeNext, strict; tsup esm+cjs dts; vitest with coverage
  thresholds; prettier printWidth 150). `@langchain/*`, `interfaze`, `zod`, `openai`
  stay `external` in tsup.

## The `ChatInterfaze` class

`src/chat_models.ts` — `class ChatInterfaze extends ChatOpenAICompletions`.
Parity map against the Python implementation:

| Python | JS |
|---|---|
| `__init__` — key from arg/`INTERFAZE_API_KEY`, else raise `InterfazeError`; default `base_url`→`INTERFAZE_BASE_URL`, `model`→`INTERFAZE_MODEL` | constructor `{ apiKey?, model?, ...fields }`: resolve `apiKey ?? process.env.INTERFAZE_API_KEY`, throw `InterfazeError` if absent; `configuration.baseURL ??= INTERFAZE_BASE_URL`; `model ??= INTERFAZE_MODEL`; forward the rest to `super` |
| `precontext` field → `extra_body.precontext` in `_get_request_payload` | accept a `precontext` field; merge `{ precontext }` into `modelKwargs` in the constructor (OpenAI JS SDK spreads unknown fields top-level, so no `invocationParams` override needed) |
| `_rewrite_video_blocks` / `_convert_video_block` in `_get_request_payload` | private `rewriteVideoBlocks(messages)`; rewrite `{type:"video", url\|base64\|file_id, mime_type?, extras.filename?}` → `{type:"file", file:{file_data\|file_id, filename?}}`; applied at the top of every send path before calling `super` |
| `_create_chat_result` — read `precontext`/`reasoning`/`vcache` off raw response, set on `response_metadata` + `additional_kwargs`; strip `<think>`/`<precontext>` tags from string content | construct with `__includeRawResponse: true`; override `_generate`: rewrite video blocks, call `super._generate`, read `additional_kwargs.__raw_response` for the three side fields, copy them onto `response_metadata` + `additional_kwargs`, run tag-stripping (`stripSideChannels`) on string content, then **delete `__raw_response`** so output matches Python's clean message |
| `_convert_chunk_to_generation_chunk` — read side fields off each chunk | inside the stream override, read side fields off each chunk's raw and attach to the chunk message |
| `_stream` / `_astream` — feed content through `SideChannelFilter`, emit a final chunk with flushed tail + reasoning/precontext | override `_streamResponseChunks` (async generator): rewrite video blocks, iterate `super._streamResponseChunks`, feed each chunk's string content through `SideChannelFilter.feed`, then after the loop emit a final `ChatGenerationChunk` carrying `filter.flush()` tail + `reasoning`/`precontext` from `stripSideChannels` over the accumulated raw |
| (JS-only trap) | **also** override `_streamChatModelEvents` — `ChatOpenAICompletions` ships a native implementation that bypasses `_streamResponseChunks`, so `.streamEvents()` would otherwise stream **unfiltered** content. Route it through the filtered chunk path (or re-apply the same filter) so all streaming surfaces are filtered, matching Python |
| `is_lc_serializable() -> False` | `is_lc_serializable()` returns `false` |
| `with_structured_output`, `bind_tools` inherited | `withStructuredOutput` (incl. `includeRaw`) and `bindTools` inherited cleanly — no override |

`src/index.ts` re-exports `ChatInterfaze` (and its options type).

### Async parity note
Python has explicit sync + async twins (`_stream`/`_astream`, invoke/ainvoke). JS is
single-implementation async throughout — `_generate` and `_streamResponseChunks` are
already async, and `.invoke`/`.stream`/`.batch` are all async in JS LangChain. So the
one `_generate` + one `_streamResponseChunks` override covers both the Python sync and
async paths.

## interfaze-js edit (separate repo)

In `/Users/mylo/Work/interfaze-js`:

1. `src/stream.ts`: add `export` to `class SideChannelFilter`.
2. `src/index.ts`: `export { InterfazeChatCompletionStream, stripSideChannels, stripJsonFence, SideChannelFilter } from "./stream.js";`
   (add `stripSideChannels` + `SideChannelFilter`; `stripJsonFence` optional).
3. Bump `package.json` version to `1.0.3`; add a CHANGELOG entry.

This mirrors interfaze-py, which already exports `SideChannelFilter` and
`strip_side_channels` for the Python langchain package to consume. This repo then
imports them from the `interfaze` npm package.

## Tests

Vitest, mirroring interfaze-js's fixture-driven style (`test/fixtures/*.json`,
recorded response bodies + SSE streams). Unit coverage:

- API key: resolves from arg, from `INTERFAZE_API_KEY`, throws `InterfazeError` when absent.
- Defaults: `baseURL` and `model` default to the Interfaze constants; user overrides win.
- `precontext` constructor field lands as a top-level `precontext` in the request body.
- Video block rewrite: `url`, `base64` (with/without `mime_type`), `file_id`, `extras.filename`;
  error when none of the three sources present.
- Side fields (`precontext`/`reasoning`/`vcache`) surface on both `response_metadata`
  and `additional_kwargs`; `__raw_response` is not leaked.
- Tag stripping: `<think>`/`<precontext>` removed from non-streaming string content.
- Streaming filter: side-channel tags stripped across chunk boundaries; final chunk
  carries reasoning/precontext.
- `.streamEvents()` is filtered too (guards the `_streamChatModelEvents` bypass).
- `withStructuredOutput` (Zod + JSON schema, `includeRaw`) and `bindTools` produce
  parsed output / `tool_calls`.

Plus `@langchain/standard-tests` (JS analog of Python's `langchain-tests`) for the
standard chat-model conformance suite.

Coverage thresholds copied from interfaze-js (lines/statements/functions 95, branches 88).

## CI / publish

- **ci.yml:** split into `python-*` and `js-*` jobs. Python jobs (lint/type/test with
  uv, mirroring the current workflow) run in `python/`; JS jobs (format check,
  typecheck, build, `check:pkg`, packed-tarball smoke test, test matrix on Node
  20/22/24) run in `js/`, mirroring interfaze-js's CI. Keep the gitleaks secret scan.
- **publish.yml:** Python → PyPI job (existing) and JS → npm + JSR jobs (mirror
  interfaze-js's `npm-publish` with `--provenance --access public` and `jsr-publish`).
  Gate each on the relevant tag/path so a release publishes the right package(s).

## Docs

- Move current root README → `python/README.md` (unchanged content).
- `js/README.md`: TS-flavored port of the same sections — install `@interfaze/langchain`,
  setup, first request (structured output from an image), precontext, chat, streaming,
  structured output, tools, reasoning, multimodal + video, async/batch, chains (LCEL
  `.pipe()`), feeding precontext, tasks/guardrails (punt to core client), errors,
  capabilities table.
- New root `README.md`: one-paragraph intro to the two integrations with links to
  `python/README.md` and `js/README.md` and to the upstream SDKs.

## Out of scope

- No reimplementation of `tasks.*` / `guard` in the langchain package — the README
  points to the core `interfaze` client, exactly as the Python README does.
- Python and JS `langchain-core` versions are intentionally **not** aligned (Python
  pins `langchain-openai <1.5` / `langchain-core >=1.0`; JS needs `@langchain/core
  ^1.2.2`). They are independent packages.

## Risks

- **Unstable LangChain surface.** The overrides subclass `ChatOpenAICompletions`
  (`@internal`) and, if used, the `@deprecated` converter hooks. Mitigation: pin the
  exact `@langchain/openai` version, prefer `__includeRawResponse` over the deprecated
  converter hook, and cover every override with a test that fails loudly on a break.
- **`.streamEvents()` bypass.** Overriding only `_streamResponseChunks` leaves
  `.streamEvents()` unfiltered. Mitigation: override `_streamChatModelEvents` too;
  explicit test.
- **`@interfaze` npm scope ownership.** Must be registered/owned before publish. Not a
  build blocker; a release blocker.
```
