# Contributing

Two packages, one repo: [`python/`](./python) (`interfaze-langchain`) and [`js/`](./js) (`@interfaze/langchain`). A change to one usually needs the same change to the other — the two are kept behaviourally identical.

## Setup

```bash
cd python && uv sync --all-groups
cd js && npm ci
```

## Unit tests

Offline — every request is mocked, and CI runs these on every push.

```bash
cd python && uv run pytest tests/unit_tests/
cd js && npm test
```

Python also runs `ruff check .`, `ruff format --check .` and `mypy`; JS runs `npm run typecheck`, `npm run format:check` and `npm run check:pkg`. CI runs the Python suite against 3.10–3.13 and the JS suite against Node 20/22/24, so check the ends of both matrices before pushing anything version-sensitive:

```bash
cd python && uv run --python 3.10 --all-groups pytest tests/unit_tests/
```

## The langchain-tests standard harness

`tests/integration_tests/` is LangChain's own `ChatModelIntegrationTests` conformance suite. It makes real calls, so it is not in CI and needs a key:

```bash
cd python
INTERFAZE_API_KEY=sk_... uv run pytest tests/integration_tests
```

## Live QA

A go/no-go gate against the real API — every modality, the streaming side channels, and the negative contract cases. Not part of PR CI; the `Live QA` workflow runs it weekly and on demand.

```bash
export INTERFAZE_API_KEY=sk_...
export INTERFAZE_BASE_URL=https://api.interfaze.ai/v1   # optional

cd python && uv run python scripts/qa_live.py
cd js && npm run qa:live
```

Run both before cutting a release. They exercise paths the mocked suites cannot: real `<think>` streaming, precontext from live tool runs, and the server-side validation limits the READMEs document.

## Releasing

Five files carry the version and must agree — `python/pyproject.toml`, `python/interfaze_langchain/_version.py`, `js/package.json`, `js/jsr.json`, `js/src/version.ts`. The last two reach users as a `User-Agent`.

```bash
node scripts/check-versions.mjs          # do the five agree?
node scripts/check-versions.mjs v1.2.3   # ...and do they match the tag?
```

CI runs the first form on every PR. Publishing runs the second against the release tag and re-runs both test suites before anything is uploaded; a GitHub prerelease goes to TestPyPI only, a full release to PyPI, npm and JSR.
