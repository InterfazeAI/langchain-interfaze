import { defineConfig } from "vitest/config";

/**
 * Live suite: real calls against the Interfaze API. Needs `INTERFAZE_API_KEY`
 * (every test is skipped without it); `INTERFAZE_BASE_URL` overrides the endpoint.
 * Run with `npm run test:live`.
 */
export default defineConfig({
  test: {
    include: ["test-live/**/*.live.test.ts"],
    environment: "node",
    testTimeout: 300_000,
    hookTimeout: 300_000,
    fileParallelism: true,
    pool: "threads",
    reporters: ["verbose"],
  },
});
