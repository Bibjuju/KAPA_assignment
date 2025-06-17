# Kapa PDF take-home

This is the kapa.ai takehome for the [Research Engineer](https://www.ycombinator.com/companies/kapa-ai/jobs/huRVu7n-research-engineer-applied-ai) position.

## Objective

Primary objective:
- We want to test your ability to quickly solve an open-ended problem. Hence these instructions are intentionally vague.

Secondary objectives:
- We are looking for evidence that you can understand and work within a code base of moderate complexity.
- We want to give you an impression of the work we do at Kapa. This is a real story we did in the past packaged into a toy app.

## Problem Motivation

Many of our customers have large amounts of PDFs. For example Nordic Semiconductors and Silicon Labs have PDF datasheets for all of the chips they make which can be hundreds of pages long.

The Kapa system runs on Markdown. This means that before we can index their PDF files we need to convert them to Markdown. This is a pretty difficult task, i.e.,

- Fixed layout vs. flowing text: PDF stores absolute glyph positions, so reconstructing a logical reading order for Markdown—especially with multi-column layouts—is non-trivial.
- Missing structural cues: Headings, paragraphs, lists, and tables are only visual in PDF, forcing you to infer Markdown syntax from font size, style, and spacing.
- Images and figures: PDFs embed graphics inline, whereas Markdown needs external image files and links, requiring extraction, naming, and referencing.
- Tables: Cell boundaries are just drawn lines, so converting them into pipe-delimited Markdown tables risks misaligned or lost structure.
- ...

This take home asks you to solve PDF to markdown conversion because it is a real past problem of ours that is very open-ended and requires some programming. 

## System Description

We have set up the following toy RAG application so you could work on this problem.

It is simple but functionally complete. A central `RAGAgent` class orchestrates both indexing of and
question answering across a set of PDF files:

```
(data/pdfs) ──► DirectoryPDFLoader
                    │
                    ▼
              PymuConverter  (PDF → Markdown)
                    │
                    ▼
        MarkdownSectionChunker  (Markdown → chunks)
                    │
                    ▼
          InMemoryVectorStore  (embeddings in RAM)
                    │
                    ▼
                 RAGAgent  (retrieval + answer generation)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Streamlit “Browse”     Streamlit “Chat”
      (inspect)          (ask questions)
```

| Path | Class | What it does |
|------|-------|--------------|
| `src/loader/pdf_loader.py` | **DirectoryPDFLoader** | Loads every PDF in `data/pdfs/` as bytes. |
| `src/converter/pymu.py` | **PymuConverter** | *Placeholder* converter using **PyMuPDF** to convert PDFs to Markdown. |
| `src/chunker/markdown_section_chunker.py` | **MarkdownSectionChunker** | Splits Markdown into semantically coherent sections. |
| `src/vector_store/in_memory.py` | **InMemoryVectorStore** | Embeds chunks with `text-embedding-3-small` and stores them in RAM. |
| `src/agent/rag_agent.py` | **RAGAgent** | Orchestrates the full pipeline shown above and lets the user ask questions about the documents. |
| `app/streamlit_app.py` | Streamlit UI | **Browse** documents/chunks or **Chat** with the RAG system. |


## Deliverables

The `PymuConverter` currently used to convert the PDFs to Markdown is a suboptimal placeholder. 
- You should improve or replace it while leaving the rest of the system unchanged. Your task is to improve the quality of this RAG system exclusively by improving its PDF to Markdown conversion. However, you will have to understand the rest of the code to arrive at a good solution.
- There are no limitations on how you are allowed to solve this problem. You can use whatever technologies you like.
- As you go along please document your thoughts and findings. Understanding how you think about this problem is equally important to us as the actual solutions you come up with. A simple markdown file will suffice for this.
- You can judge the quality of your conversion by trying the `Example Question-Answer Pairs` below

### Example Question–Answer Pairs

Use the tables below to gauge progress after you replace `PymuConverter`.

* **Status column**  
  * **✓ Works** – the baseline converter already returns the correct information.  
  * **✗ Needs improvement** – the baseline fails; your converter should fix this.
* **Correct answer** – wording may differ; the answer simply needs to contain
  the same factual content.
* **Page** – PDF page on which the answer can be found.

### `21098-ESPS2WROOM-scan.pdf`

| Status | Question | Correct answer (information only) | Page |
|--------|----------|-----------------------------------|------|
| ✗ Needs improvement | What type of equipment is **B20111311**? | Modular Approval, Wi-Fi Device | — |
| ✗ Needs improvement | When was the certificate for **US0057** issued? | 2020-11-19 | — |
| ✗ Needs improvement | Who holds the **21098-ESPS2WROOM** certificate? | ESPRESSIF SYSTEMS (SHANGHAI) CO., LTD. | — |

### `esp8266_hardware_design_guidelines_en.pdf`

| Status | Question | Correct answer (information only) | Page |
|--------|----------|-----------------------------------|------|
| ✓ Works | Can **ESP8266EX** be applied to any micro-controller design as a Wi-Fi adaptor? | Yes; via SPI/SDIO or I2C/UART interfaces | 6 |
| ✗ Needs improvement | What is the **frequency range** for ESP8266EX? | 2.4 G – 2.5 G (2400 M – 2483.5 M) | 7 |
| ✗ Needs improvement | To what pin do I connect the **resistor** for ESP8266EX? | Pin ERS12K (31) | 15 |

### `esp8266-technical_reference_en.pdf`

| Status | Question | Correct answer (information only) | Page |
|--------|----------|-----------------------------------|------|
| ✓ Works | What’s the **flash memory** of EFM8BB31F32G-D-QFP32? | 32 kB | 4 |
| ✓ Works | What is the **maximum storage temperature** for EFM8BB3? | 150 °C | 40 |
| ✗ Needs improvement | How many **multi-function I/O pins** does EFM8BB3 have? | Up to 29 | 10 |
| ✗ Needs improvement | What is the **minimum Voltage Reference Range for DACs**? | 1.15 V | 31 |
| ✗ Needs improvement | What are the different **power modes** for EFM8BB3? | Normal, Idle, Suspend, Stop, Snooze, Shutdown | 10 |

**Goal:** After implementing your improved converter, most—ideally all—rows
currently marked **Needs improvement** should return the correct information.

## Set up

To get started you can follow these steps:

### Prerequisites

* **Docker**    
    with the Compose plugin (standard in Docker Desktop).
* **Visual Studio Code**    
    with the **Docker** extension installed. (Optional)
* **OpenAI API key**    
    Create a `.env` file in the project root and add a real OpenAI API key.

  ```text
  OPENAI_API_KEY=<your-key-here>
  ```

This repo only uses `gpt-4.1-mini-2025-04-14` and `text-embedding-3-small` so the cost should be negligible.

### Getting started (VSCode)

1. **Open the project**  
   After un‑zipping the assignment archive, choose **File → Open Folder…** in VS Code and select the extracted directory.

2. **Build the container**  
   Open the **Docker** side bar, right‑click `compose.yaml` under *Compose* and select **Compose Up**.  
   The image is built and the container starts, but Streamlit is not yet running.

4. **Attach a shell to the container**  
    In the VS Code Docker side bar:
    right‑click the running `streamlit_app` container → Attach Shell

3. **Launch Streamlit inside the container**    
   Inside the shell:

   ```bash
   make run 
   ```
   go to **http://localhost:8501** to use the UI.

5. **Index and explore**
   * In the Streamlit sidebar click **Load & index PDFs**.  
   * Use **Browse** to inspect the converted Markdown and its chunks. This will help you understand the behavior of your converter.
   * Switch to **Chat** and try the example questions.

The `app/`, `src/`, and `data/` directories are volume‑mounted, so edits you make in VS Code are reflected immediately inside the running container.

Of course you can do this with any other editor besides VSCode or simply from the command line. These steps are just meant to illustrate the general workflow.

## Submission

When you are done with your work, create a private GitHub repository, open a pull request, and invite the following reviewers:

- finn@kapa.ai
- janis@kapa.ai
- frederikdieleman@kapa.ai

Please ensure the repository contains both your solution(s) and a clear record of your thinking process.

If the final converter isn’t perfect, submit what you have.
We are equally interested in how you reasoned about trade-offs, which ideas you tried, why you rejected certain paths, and what you would attempt next. A well-documented “journey” often tells us more about your skills than a flawless end result.

## Giftcard

If you submit a solution to us you get a USD 300 Amazon gift card as we really value your time investment.