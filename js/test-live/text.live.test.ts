import { describe, expect, it } from "vitest";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { chat, FAST, freshChat, hasKey, lower, SLOW, text } from "./helpers.js";

describe.skipIf(!hasKey)("text", () => {
  const llm = chat();

  it("text_gen_story", { timeout: FAST }, async () => {
    const res = await llm.invoke("Write a short story about a robot learning to paint");
    expect(text(res).length).toBeGreaterThan(50);
  });

  it("text_gen_story with a system message", { timeout: FAST }, async () => {
    const res = await llm.invoke([
      new SystemMessage("You are a helpful assistant."),
      new HumanMessage("Write a short story about a robot learning to paint"),
    ]);
    expect(text(res).length).toBeGreaterThan(50);
  });

  it("text_capital", { timeout: FAST }, async () => {
    const res = await llm.invoke("What is the capital of France? Answer in one word.");
    expect(lower(res.content)).toContain("paris");
  });

  it("surfaces the interfaze envelope on every response", { timeout: FAST }, async () => {
    const res = await llm.invoke("Hello");
    expect(res.response_metadata.model_provider).toBe("interfaze");
    expect(typeof res.response_metadata.vcache).toBe("boolean");
    expect(res.usage_metadata?.total_tokens).toBeGreaterThan(0);
  });
});

describe.skipIf(!hasKey)("structured output", () => {
  const llm = chat();

  it("structured_weather", { timeout: FAST }, async () => {
    const schema = z.object({ city: z.string(), temperature_celsius: z.number(), condition: z.string() });
    const out = await llm.withStructuredOutput(schema, { name: "weather_schema" }).invoke("What is the current weather in Tokyo?");
    expect(out.city.length).toBeGreaterThan(0);
    expect(typeof out.temperature_celsius).toBe("number");
    expect(out.condition.length).toBeGreaterThan(0);
  });

  it("structured_founder", { timeout: FAST }, async () => {
    const out = await llm.withStructuredOutput(z.object({ name: z.string() })).invoke("Who is the founder of JigsawStack?");
    expect(out.name.length).toBeGreaterThan(0);
  });

  it("structured_capital_pop", { timeout: FAST }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ city: z.string(), population_millions: z.number() }))
      .invoke("What is the capital of France and its approximate metro population in millions?");
    expect(lower(out.city)).toContain("paris");
    expect(typeof out.population_millions).toBe("number");
  });

  it("structured_json_no_fences", { timeout: FAST }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ capital: z.string() }), { includeRaw: true })
      .invoke("Return ONLY a JSON object (no markdown fences) with key 'capital' set to the capital of France.");
    expect(text(out.raw as never)).not.toContain("```");
    expect(lower((out.parsed as { capital: string }).capital)).toContain("paris");
  });
});

describe.skipIf(!hasKey)("reasoning", () => {
  // The semantic cache replays a stored answer without its <think> block.
  const llm = freshChat();

  it("reasoning_math", { timeout: FAST }, async () => {
    const res = await llm.invoke("What is 25 * 47?", { reasoningEffort: "high" });
    expect(text(res)).toContain("1175");
    expect(String(res.response_metadata.reasoning).length).toBeGreaterThan(0);
    expect(text(res)).not.toContain("<think>");
  });
});

describe.skipIf(!hasKey)("runnable surface", () => {
  const llm = chat();

  it("composes in an LCEL chain", { timeout: FAST }, async () => {
    const chain = ChatPromptTemplate.fromTemplate("Translate to {lang}: {text}").pipe(llm).pipe(new StringOutputParser());
    const out = await chain.invoke({ lang: "French", text: "Hello" });
    expect(out.length).toBeGreaterThan(0);
  });

  it("batches concurrently", { timeout: SLOW }, async () => {
    const out = await llm.batch(["Summarize the colour blue in one sentence.", "Name one planet.", "What is 2+2?"]);
    expect(out).toHaveLength(3);
    expect(out.every((m) => text(m).length > 0)).toBe(true);
  });
});
