---
description: Ground HLS code edits in the indexed Xilinx Vitis HLS User Guide via the rag-memory MCP. Use whenever about to add, remove, or modify a `#pragma HLS ...` directive, a `config_interface` / `config_rtl` / other TCL setting in a Synthesis script, when diagnosing a synthesis-report symptom (Widen Fail, II Violation, Burst inferred/failed, negative slack, DATAFLOW issue, FIFO depth), when choosing between alternative HLS mechanisms, or when the user asks a "how do I" / "what does X do" question about HLS. Triggers on edits to HLS C++ (none left in kernels/: the C++ reference models in kernels/*_rtl/model/ carry no pragmas), *.tcl.in, *.tcl, and any HLS-related discussion (II, pipelining, dataflow, AXI widening, burst inference, m_axi, line buffers).
allowed-tools: Read Edit Write Bash mcp__rag-memory__hybridSearch mcp__rag-memory__listDocuments mcp__rag-memory__getDetailedContext mcp__rag-memory__searchNodes mcp__rag-memory__openNodes mcp__rag-memory__extractTerms mcp__rag-memory__readGraph mcp__rag-memory__getKnowledgeGraphStats
---

# hls-rag

Retrieval-augmented helper for HLS code. The `rag-memory` MCP indexes a small library of Xilinx PDFs; the primary one for kernel work is **`docs/ug1399-vitis-hls-en-us-2025.2.pdf`** (Vitis HLS User Guide v2025.2, 903 pages, January 2026 — same major version as the project's installed Vitis HLS). Supporting docs cover the Zynq UltraScale+ PS (PG201), the PS-PL VIP (DS941), and an embedded-design tutorial (UG1209).

Use this skill to **look up the authoritative answer before writing or recommending HLS code**, instead of relying on training-set memory of pragma syntax or scheduling rules.

## When to query the MCP

Run a `mcp__rag-memory__hybridSearch` *before* committing to an answer or an edit in any of these situations:

- **Adding / removing / changing a `#pragma HLS …` directive.** Look up the canonical syntax, valid arguments, and the documented effect.
- **Editing TCL synthesis settings** (`config_interface`, `config_rtl`, `config_compile`, `set_clock_uncertainty`, `set_directive_*`, etc.) — confirm the option exists in 2025.2 and what it controls.
- **Diagnosing a synthesis-report symptom**: `Widen Fail`, `Could not widen since type i…`, `Inferred burst reverted`, `II Violation - Resource Limitation`, `Stride is incompatible`, "II of 2 due to memory port", negative slack on a specific stage, DATAFLOW deadlock or scheduling errors, FIFO depth chosen wrong.
- **Choosing between alternative mechanisms**: `STREAM` vs `PIPO`, `AGGREGATE` vs `ARRAY_RESHAPE`, `BIND_STORAGE` vs `RESOURCE` (deprecated), `LATENCY min/max` vs manually splitting a stage, etc.
- **The user asks a how-to question**: "how do I make HLS widen the bus", "how do I tell HLS this pointer is aligned", "why is HLS not pipelining this loop", etc.

**Skip the lookup** only for purely mechanical tasks (typo fixes, variable renames, formatting) or for project-specific knowledge that the docs can't answer (CMake structure, existing kernel architecture).

## How to query

1. **Pick a natural-language query** that names the concept and the suspected mechanism. The hybrid search combines vector similarity + knowledge-graph traversal, so it does best with full phrases like:
   - `"m_axi automatic port width widening alignment conditions"`
   - `"DATAFLOW preconditions sub-function inline limits"`
   - `"set_clock_uncertainty effect on schedule slack"`
   - `"BIND_STORAGE ram_t2p dual-port BRAM"`

2. **Call `mcp__rag-memory__hybridSearch`** with that query. Defaults are fine; bump `limit` to 5–7 when results look thin or the question spans several pragmas. Leave `useGraph=true` so the related-entity boost kicks in.

3. **Read the `key_highlight` + `content_summary` fields** in the response — they're the matched passages. Each result lists:
   - `relevance_score` — overall hybrid rank
   - `document_title` — prefer `"Vitis HLS User Guide"` chunks; PG201 / DS941 chunks only when the question touches the PS-PL boundary
   - `chunk_id` — opaque id, useful for `getDetailedContext` follow-up
   - `entities` — graph nodes touched (often gives you the canonical pragma/TCL command names)

4. **Follow up if needed:**
   - `mcp__rag-memory__getDetailedContext({chunkId: "<id>"})` for the full chunk text when the highlight isn't enough.
   - `mcp__rag-memory__searchNodes({query: "pragma HLS dataflow"})` → `openNodes` to fetch a specific pragma's entity (with its observations and relationships) — handy when the user names a pragma explicitly.

5. **Cross-check** with another query if the first one returned PG201/DS941 chunks instead of UG1399 — rephrase to use HLS terms (`pragma`, `pipeline`, `II`, `m_axi`) that pull the right doc.

## How to ground the answer

When writing code or replying:

- **Quote the exact syntax** from the retrieved chunk. Don't paraphrase pragma arguments from memory.
- **Cite the chunk** inline: e.g. *"UG1399, chunk `ug1399-…_chunk_7`: 'the start of the sequential accesses needs to be aligned to the widen word size'"*. The user can verify.
- **If the doc says "the tool does X"**, treat it as authoritative and follow it — don't override based on a hunch.
- **If the doc is silent** on a specific concern, say so explicitly (*"UG1399 doesn't document this case — falling back to the kernel's existing convention."*).
- **If two chunks contradict each other** (rare — usually older doc text in a different appendix), prefer the more recent / more specific one, and surface the contradiction to the user.

## What NOT to do

- **Don't mutate the knowledge base.** Read-only tools only. Write operations (`chunkDocument`, `storeDocument`, `createEntities`, `addObservations`, any `delete*`, `embedAllEntities`, `embedChunks`, `linkEntitiesToDocument`, `createRelations`, `deleteRelations`) are deliberately omitted from `allowed-tools`.
- **Don't WebFetch / WebSearch when the question is plausibly in UG1399.** The indexed v2025.2 doc matches the project's installed Vitis HLS — web sources may describe 2022.x or 2024.x syntax which has subtly different option names.
- **Don't skip the lookup to "save time".** A 1-shot `hybridSearch` is faster than a failed edit + re-synthesis cycle, and the cited answer is verifiable.

## Quick-reference: what's indexed

| Document | Use for |
|---|---|
| `docs/ug1399-vitis-hls-en-us-2025.2.pdf` — Vitis HLS User Guide v2025.2 | **Primary.** All HLS pragmas, TCL commands, scheduling, m_axi/s_axilite/axis behavior, burst inference, port widening, DATAFLOW, ARRAY_PARTITION, etc. |
| `docs/pg201-zynq-ultrascale-plus-processing-system_2020.pdf` — PG201 | PS-PL AXI port specs (HPM/HPC widths, IRQ_F2P, FCLK, DDR, PS Configuration Wizard). |
| `docs/ds941-zynq-ultra-ps-e-vip.pdf` — DS941 | PS-VIP for SystemVerilog testbenches (the one used in `hw/cormorant_test_stand`). |
| `docs/ug1209-embedded-design-tutorial.pdf` — UG1209 | ZCU102 boot / PetaLinux / FSBL / XMPU-XPPU — only relevant to on-board bring-up. |

Knowledge-graph entity types worth knowing:
- **`HLS_PRAGMA`** (9 entries) — `pragma HLS interface`, `array_partition`, `array_reshape`, `pipeline`, `dataflow`, `stream`, `unroll`, `dependence`, `loop_flatten`.
- **`HLS_TCL_COMMAND`** (3 entries) — `csynth_design`, `cosim_design`, `export_design`.
- **`CONCEPT`** (17 entries) — overarching HLS topics; queryable via `searchNodes`.

Use `mcp__rag-memory__getKnowledgeGraphStats` once at the start of a long HLS session to confirm the MCP is alive and the counts haven't drifted.

## Worked example

User: *"HLS isn't widening my m_axi gmem1 to 32 bits even though max_widen_bitwidth is 32."*

1. `mcp__rag-memory__hybridSearch({query: "m_axi automatic port width widening conditions alignment iteration count", limit: 5})`.
2. Top result: UG1399 chunk on Port Width Resizing. Highlight quotes the preconditions: *"The start of the sequential accesses needs to be aligned to the widen word size … the length of the sequential accesses needs to be divisible by the widen factor … If the size and number of iterations are variable at compile time, then the tool will not automatically widen port widths."*
3. Tell the user: HLS won't auto-widen because their iteration counts come from runtime AXI-Lite registers — the guide explicitly says so. Point them at the doc-suggested escape hatch: *"you can manually change the port width by using Vector Data Types or Arbitrary Precision (AP) Data Types as the data type of the port."*
4. Cite the chunk_id alongside the answer.
5. Only then write code (e.g. `ap_uint<32>*` reinterpret cast) — with the doc's justification on record.

That same workflow applies to every HLS question: search → read → quote → edit.
