import { describe, expect, it } from "vitest";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import { chat, filePart, FILES, hasKey, IMAGES, imagePart, lower, precontextNames, receiptB64, SLOW } from "./helpers.js";

const llm = chat();

function ask(prompt: string, part: Record<string, unknown>): HumanMessage {
  return new HumanMessage({ content: [{ type: "text", text: prompt }, part] as never });
}

const bbox = {
  top_left_x: z.number(),
  top_left_y: z.number(),
  bottom_right_x: z.number(),
  bottom_right_y: z.number(),
};

describe.skipIf(!hasKey)("vision / ocr", () => {
  it("ocr_id_document", { timeout: SLOW }, async () => {
    const schema = z.object({
      full_first_name: z.string(),
      full_last_name: z.string(),
      full_address: z.string().nullable(),
      email: z.string().nullable(),
      id_type: z.string(),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("Extract information from the image based on the schema.", imagePart(IMAGES.idMedium))]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(lower(parsed.full_first_name)).toContain("iv");
    expect(lower(parsed.full_last_name)).toMatch(/mu(ñ|n)oz/);
    expect(parsed.id_type.length).toBeGreaterThan(0);
    expect(precontextNames(out.raw as never)).toContain("ocr");
  });

  it("ocr_id_jpg", { timeout: SLOW }, async () => {
    const schema = z.object({
      first_name: z.string(),
      last_name: z.string(),
      dob: z.string(),
      driver_licence_number: z.string(),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("Extract the details from this ID", imagePart(IMAGES.idJpg))]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.first_name.length).toBeGreaterThan(0);
    expect(parsed.dob.length).toBeGreaterThan(0);
    expect(parsed.driver_licence_number.length).toBeGreaterThan(0);
    expect(precontextNames(out.raw as never)).toContain("ocr");
  });

  it("ocr_receipt_fields", { timeout: SLOW }, async () => {
    const item = z.object({ name: z.string(), price: z.string() });
    const schema = z.object({
      items: z.array(item),
      highlighted_items: z.array(item),
      total_cost: z.string(),
      tax: z.string(),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("Extract text from the image.", imagePart(IMAGES.receiptItems))]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.items.length).toBeGreaterThan(0);
    expect(parsed.total_cost).toContain("144.02");
    expect(parsed.tax).toContain("4.58");
    expect(lower(parsed.highlighted_items.map((i) => i.name).join(" "))).toContain("gale");
    expect(precontextNames(out.raw as never)).toContain("ocr");
  });

  it("ocr_store_location", { timeout: SLOW }, async () => {
    const schema = z.object({
      query_results: z.array(z.object({ text: z.string(), confidence: z.number() })),
      text: z.string(),
      confidence: z.number(),
    });
    const out = await llm.withStructuredOutput(schema).invoke([ask("Where is this store located?", imagePart(IMAGES.receipt))]);
    const haystack = lower(`${out.query_results.map((r) => r.text).join(" ")} ${out.text}`);
    expect(haystack).toContain("greenwood");
  });

  it("ocr_word_bboxes", { timeout: SLOW }, async () => {
    const schema = z.object({
      text: z.string(),
      words: z.array(z.object({ text: z.string(), ...bbox })),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("extract every word and its position from this receipt", imagePart(IMAGES.receipt))]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.words.length).toBeGreaterThanOrEqual(10);
    for (const w of parsed.words) {
      expect(w.top_left_x).toBeGreaterThanOrEqual(0);
      expect(w.top_left_y).toBeGreaterThanOrEqual(0);
      expect(w.bottom_right_x).toBeGreaterThanOrEqual(w.top_left_x - 2);
      expect(w.bottom_right_y).toBeGreaterThanOrEqual(w.top_left_y - 2);
    }
    expect(precontextNames(out.raw as never)).toContain("ocr");
  });

  it("ocr_pdf_title_authors", { timeout: SLOW }, async () => {
    const schema = z.object({ title: z.string(), authors: z.array(z.string()).min(1) });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("Extract the title and author names from the first page.", filePart(FILES.attentionPdf, "attention.pdf"))]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(lower(parsed.title)).toContain("attention");
    expect(lower(parsed.authors.join(" "))).toContain("vaswani");
    expect(precontextNames(out.raw as never)).toContain("ocr");
  });

  it("ocr_multilang", { timeout: SLOW }, async () => {
    const schema = z.object({
      translations: z.array(
        z.object({
          text_in_original_language: z.string(),
          text_in_telugu: z.string(),
          width_of_image: z.number(),
          height_of_image: z.number(),
        })
      ),
    });
    const out = await llm
      .withStructuredOutput(schema)
      .invoke([ask("Extract information from the image based on the schema.", imagePart(IMAGES.multilang))]);
    expect(out.translations.length).toBeGreaterThan(0);
  });

  it("ocr_document_layout", { timeout: SLOW }, async () => {
    const element = z.object({
      type: z.enum(["heading", "paragraph", "formula", "figure", "table", "caption", "list"]),
      content: z.string(),
      ...bbox,
    });
    const schema = z.object({
      pages: z.array(z.object({ page_number: z.number(), elements: z.array(element) })),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([
        ask(
          "extract the layout elements (headings, paragraphs, figures, tables) with bounding boxes from the first page",
          filePart(FILES.attentionPdf, "attention.pdf")
        ),
      ]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.pages.length).toBeGreaterThan(0);
    const elements = parsed.pages.flatMap((p) => p.elements);
    expect(elements.length).toBeGreaterThan(0);
    expect(elements.some((e) => e.type === "heading" || e.type === "paragraph")).toBe(true);
    expect(precontextNames(out.raw as never)).toContain("ocr");
  });
});

describe.skipIf(!hasKey)("document extraction / markdown", () => {
  const inlineReceipt = () => imagePart(`data:image/jpeg;base64,${receiptB64()}`);

  it("doc_invoice_exact", { timeout: SLOW }, async () => {
    const schema = z.object({
      vendor_name: z.string(),
      total_amount: z.number(),
      bill_date: z.string(),
      line_items: z.array(z.object({ description: z.string(), price: z.number().nullable() })),
    });
    const out = await llm
      .withStructuredOutput(schema, { name: "bill" })
      .invoke([
        ask(
          "Extract the bill data from this receipt: vendor_name, total_amount (number), bill_date (YYYY-MM-DD), and line_items[{description, price}].",
          inlineReceipt()
        ),
      ]);
    expect(Math.abs(out.total_amount - 15.15)).toBeLessThanOrEqual(0.01);
    expect(out.bill_date).toBe("2018-05-06");
    expect(lower(out.vendor_name)).toContain("marco polo");
    expect(out.line_items.length).toBeGreaterThanOrEqual(1);
  });

  it("md_image_to_md", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ markdown: z.string() }))
      .invoke([ask("Convert this receipt image to markdown.", inlineReceipt())]);
    expect(out.markdown.length).toBeGreaterThanOrEqual(80);
    expect(lower(out.markdown)).toContain("marco polo");
    expect(lower(out.markdown)).toContain("mocha");
    expect(out.markdown).toContain("15.15");
  });

  it("md_pdf_to_md", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ markdown: z.string() }), { includeRaw: true })
      .invoke([ask("Convert the first page of this document to markdown.", filePart(FILES.attentionPdf, "attention.pdf"))]);
    // Heading style is stable; bold emphasis is not, so it is not asserted.
    const md = (out.parsed as { markdown: string }).markdown;
    expect(md).toMatch(/#{1,3}\s*attention is all you need/i);
    expect(md.length).toBeGreaterThan(200);
    expect(precontextNames(out.raw as never)).toContain("ocr");
  });
});

describe.skipIf(!hasKey)("object / gui detection", () => {
  it("object_detection_absent", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ objects: z.array(z.object({ name: z.string() })) }))
      .invoke([ask("detect elephants in this image", imagePart(IMAGES.katana))]);
    expect(out.objects).toHaveLength(0);
  });

  it("object_detection_bbox", { timeout: SLOW }, async () => {
    const schema = z.object({ objects: z.array(z.object({ name: z.string(), ...bbox })) });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("detect the position of the katana in this image", imagePart(IMAGES.katana))]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.objects.length).toBeGreaterThan(0);
    expect(lower(parsed.objects[0]!.name)).toContain("katana");
    const box = parsed.objects[0]!;
    expect(Math.abs(box.top_left_x - 1078)).toBeLessThanOrEqual(15);
    expect(Math.abs(box.top_left_y - 474)).toBeLessThanOrEqual(15);
    expect(Math.abs(box.bottom_right_x - 1188)).toBeLessThanOrEqual(15);
    expect(Math.abs(box.bottom_right_y - 1026)).toBeLessThanOrEqual(15);
    expect(precontextNames(out.raw as never)).toContain("object_detection");
  });

  it("object_detection_multi", { timeout: SLOW }, async () => {
    const schema = z.object({ objects: z.array(z.object({ name: z.string(), ...bbox })) });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("detect all objects with bounding boxes", imagePart(IMAGES.bus))]);
    // Recall on this image swings from 1 to 8 objects run to run, and the labels vary
    // ("bus" vs a bare "object") — reproduced identically through the core interfaze SDK.
    // What the integration owns is the round-trip: well-formed boxes and the precontext.
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.objects.length).toBeGreaterThan(0);
    for (const o of parsed.objects) {
      expect(o.bottom_right_x).toBeGreaterThanOrEqual(o.top_left_x);
      expect(o.bottom_right_y).toBeGreaterThanOrEqual(o.top_left_y);
    }
    expect(precontextNames(out.raw as never)).toContain("object_detection");
  });

  it("gui_detection_form", { timeout: SLOW }, async () => {
    const schema = z.object({ gui_elements: z.array(z.object({ name: z.string(), ...bbox })) });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("find all the text fields and the clear form button", imagePart(IMAGES.guiForm))]);
    // Element count swings between 1 and 12 across runs (reproduced with the core
    // interfaze SDK), so assert the envelope rather than a per-field inventory.
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.gui_elements.length).toBeGreaterThanOrEqual(1);
    for (const e of parsed.gui_elements) {
      expect(e.bottom_right_x).toBeLessThanOrEqual(3600);
      expect(e.bottom_right_y).toBeLessThanOrEqual(2338);
    }
    expect(precontextNames(out.raw as never)).toContain("gui_detection");
  });

  it("object_detection_with_text", { timeout: SLOW }, async () => {
    const schema = z.object({
      objects: z.array(z.object({ name: z.string(), ...bbox })),
      texts: z.array(z.object({ text: z.string(), ...bbox })),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("Get the position of the crane in the image and any text", imagePart(IMAGES.construction))]);
    // The crane is not always detected; what must hold is that detection + OCR ran and
    // the schema came back well-formed.
    expect(Array.isArray((out.parsed as z.infer<typeof schema>).objects)).toBe(true);
    expect(precontextNames(out.raw as never).length).toBeGreaterThan(0);
  });
});
