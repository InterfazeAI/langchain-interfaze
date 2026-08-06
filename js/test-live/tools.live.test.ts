import { describe, expect, it } from "vitest";
import { z } from "zod";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { chat, hasKey, lower, precontextNames, SLOW, text } from "./helpers.js";

const llm = chat();

const LONG_TEXT =
  "Interfaze is a new kind of AI platform built specifically for deterministic, developer-grade tasks. " +
  "Unlike general-purpose large language models that excel at open-ended conversation but struggle with " +
  "consistency, Interfaze focuses on the operations that real software systems depend on: extracting fields " +
  "from documents, scraping structured data from arbitrary websites, transcribing audio with speaker labels, " +
  "translating content across hundreds of languages while preserving meaning, detecting objects in images and " +
  "GUI screenshots, forecasting time series without per-customer model training, and executing code in a " +
  "sandboxed environment. Every capability is exposed through an OpenAI-compatible chat completions API so " +
  "existing tooling works without modification.";

const SERIES = [
  { date: "2024-01-01", value: 412 },
  { date: "2024-01-08", value: 387 },
  { date: "2024-01-15", value: 524 },
  { date: "2024-01-22", value: 461 },
  { date: "2024-01-29", value: 398 },
  { date: "2024-02-05", value: 542 },
  { date: "2024-02-12", value: 475 },
  { date: "2024-02-19", value: 401 },
  { date: "2024-02-26", value: 558 },
  { date: "2024-03-04", value: 489 },
  { date: "2024-03-11", value: 419 },
  { date: "2024-03-18", value: 571 },
];

describe.skipIf(!hasKey)("translation", () => {
  it("translate_structured", { timeout: SLOW }, async () => {
    const schema = z.object({
      translated_text: z.string(),
      translated_text_iso_code: z.string(),
      original_text_iso_code: z.string(),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke(
        "Translate the following text into French: 'The UK drinks about 100-160 million cups of tea every day, and 98% of tea drinkers add milk to their tea.'"
      );
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(lower(parsed.translated_text_iso_code)).toContain("fr");
    expect(lower(parsed.original_text_iso_code)).toContain("en");
    expect(precontextNames(out.raw as never)).toContain("translate");
  });

  it("translate_es", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ translated_text: z.string(), target_language: z.string() }))
      .invoke("Hello, how are you today? I would like to order a coffee. — in Spanish please");
    expect(lower(out.translated_text)).toMatch(/hola|cómo|está|café/);
  });

  it("translate_long_fr", { timeout: SLOW }, async () => {
    const out = await llm.withStructuredOutput(z.object({ translated_text: z.string() })).invoke(`Can you give me this in French? "${LONG_TEXT}"`);
    expect(out.translated_text.length).toBeGreaterThanOrEqual(LONG_TEXT.length * 0.5);
    expect([" le ", " la ", " les ", " des ", " une ", " est ", " pour ", " avec "].some((w) => lower(out.translated_text).includes(w))).toBe(true);
  });

  it("translate_ja", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ translated_text: z.string() }))
      .invoke("how do you say 'Thank you for your help' in Japanese?");
    expect(/[぀-ヿ一-鿿]/.test(out.translated_text)).toBe(true);
  });

  it("translate_markup preserves html tags", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ translated_text: z.string() }))
      .invoke('Translate this to French, preserving all HTML tags exactly: Click <a href="/start">here</a> to <b>continue</b>.');
    expect(out.translated_text).toContain('<a href="/start">');
    expect(out.translated_text).toContain("</a>");
    expect(out.translated_text).toContain("<b>");
    expect(out.translated_text).toContain("</b>");
  });
});

describe.skipIf(!hasKey)("web search", () => {
  it("web_search_basic", { timeout: SLOW }, async () => {
    const res = await llm.invoke("Latest news on Nvidia");
    expect(text(res).length).toBeGreaterThan(0);
  });

  it("web_search_factual", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ founders: z.array(z.string()), year_founded: z.number(), sources: z.array(z.string()) }))
      .invoke("who founded Tesla and when?");
    expect(lower(out.founders.join(" "))).toMatch(/musk|eberhard|tarpenning/);
    expect(out.year_founded).toBe(2003);
  });

  it("web_search_history", { timeout: SLOW }, async () => {
    const schema = z.object({
      summary: z.string(),
      year: z.number(),
      month: z.string(),
      sources: z.array(z.string()).min(1),
    });
    const out = await llm.withStructuredOutput(schema).invoke("when did Apollo 11 land on the moon? give sources.");
    expect(out.year).toBe(1969);
    expect(lower(out.month)).toContain("jul");
    expect(out.summary.length).toBeGreaterThan(20);
  });

  it("web_search_structured", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ summary: z.string(), current_stock_price: z.number(), links: z.array(z.string()) }))
      .invoke("Latest news on Nvidia");
    expect(out.summary.length).toBeGreaterThan(0);
    expect(out.links.length).toBeGreaterThan(0);
  });

  it("web_search_person", { timeout: SLOW }, async () => {
    const schema = z.object({
      summary: z.string(),
      company: z.string().nullable(),
      emails: z.array(z.string()).nullable(),
      location: z.string().nullable(),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke("Who is Yoeven D Khemlani, his company, his email and where is he based now?");
    expect((out.parsed as z.infer<typeof schema>).summary.length).toBeGreaterThan(0);
    expect(precontextNames(out.raw as never).length).toBeGreaterThan(0);
  });
});

describe.skipIf(!hasKey)("scraping", () => {
  it("scrape_ecommerce", { timeout: SLOW }, async () => {
    const schema = z.object({
      products: z.array(
        z.object({
          price: z.number(),
          listing_name: z.string(),
          seller_name: z.string(),
          possible_delivery_time: z.string(),
        })
      ),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke("get all prices and listing of products for nintendo switch from https://www.amazon.com/s?k=nintendo+switch+console");
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.products.length).toBeGreaterThan(0);
    expect(parsed.products[0]!.listing_name.length).toBeGreaterThan(0);
    expect(precontextNames(out.raw as never).join(" ")).toMatch(/web_extract|search|scraper/);
  });

  it("scrape_hn", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ posts: z.array(z.object({ title: z.string(), points: z.number() })) }))
      .invoke("Extract post titles and points from https://news.ycombinator.com");
    expect(out.posts.length).toBeGreaterThan(0);
  });
});

describe.skipIf(!hasKey)("forecast", () => {
  it("forecast_series", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ predictions: z.array(z.object({ value: z.number() })) }), { includeRaw: true })
      .invoke(`Here's our weekly sales for the past 12 weeks: ${JSON.stringify(SERIES)}. What can we expect over the next month?`);
    const parsed = out.parsed as { predictions: { value: number }[] };
    expect(parsed.predictions.length).toBeGreaterThanOrEqual(3);
    for (const p of parsed.predictions) {
      expect(Number.isFinite(p.value)).toBe(true);
      expect(p.value).toBeGreaterThanOrEqual(0);
      expect(p.value).toBeLessThanOrEqual(10_000);
    }
    expect(precontextNames(out.raw as never)).toContain("forecast");
  });
});

describe.skipIf(!hasKey)("code sandbox", () => {
  it("sandbox_factorial", { timeout: SLOW }, async () => {
    const out = await llm.withStructuredOutput(z.object({ fractional: z.number() }), { includeRaw: true }).invoke("What is the factorial of 5?");
    expect((out.parsed as { fractional: number }).fractional).toBe(120);
    // The router only reaches for the sandbox when it doesn't already know the answer.
    const names = precontextNames(out.raw as never);
    if (names.length) expect(names.join(" ")).toMatch(/code_execute|code_generation/);
  });

  it("sandbox_count_r", { timeout: SLOW }, async () => {
    const out = await llm.withStructuredOutput(z.object({ answer: z.number().int() })).invoke("How many r's are there in strawberry?");
    expect(out.answer).toBe(3);
  });

  it("sandbox_codegen", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ code: z.string(), sample_input: z.string(), sample_output: z.string() }))
      .invoke("write a python script for getting cpu type using subprocess module and verify your output");
    expect(out.code).toContain("subprocess");
  });
});

describe.skipIf(!hasKey)("function calling", () => {
  it("fc_horoscope round-trips a tool result", { timeout: SLOW }, async () => {
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "get_horoscope",
          description: "Get today's horoscope for an astrological sign.",
          parameters: {
            type: "object",
            properties: { sign: { type: "string" } },
            required: ["sign"],
          },
        },
      },
    ];
    const bound = llm.bindTools(tools);
    const first = (await bound.invoke([new HumanMessage("Get my horoscope for Taurus")])) as AIMessage;
    expect(first.tool_calls?.[0]?.name).toBe("get_horoscope");

    const call = first.tool_calls![0]!;
    const second = await bound.invoke([
      new HumanMessage("Get my horoscope for Taurus"),
      first,
      new ToolMessage({ tool_call_id: call.id!, content: "Today's horoscope for Taurus: You will have a great day!" }),
    ]);
    expect(text(second).length).toBeGreaterThan(0);
  });
});
