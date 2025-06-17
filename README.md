# PDF-to-Markdown Converter

> **One-pass pipeline that turns any PDF into clean, GitHub-flavoured Markdown.**  
> Powered by PyMuPDF for rasterisation, YOLO-v10 for layout detection, Microsoft
> Table-Transformer for table structure, and Doctr for high-accuracy OCR.

---

## Key features
| Stage | Model / Library | What it does |
|-------|-----------------|--------------|
| **Rendering** | PyMuPDF | Rasterises pages at configurable zoom |
| **Layout detection** | YOLO-v10 | Finds text blocks, figures, tables, captions … |
| **Table parsing** | Table-Transformer + rule-based grid builder | Builds a 2-D cell map and emits markdown tables |
| **OCR** | Doctr (GPU/CPU) | Reads any textual crop that isn’t a table |
| **Concurrency** | `ThreadPoolExecutor` | Renders pages & crops in parallel; GPU work is batched |

* Deterministic – every run of the same PDF → identical Markdown  

---

## Repository layout of changes
```text
src/
├─ converter/
│  ├─ pymu.py          ← high-level driver
│  └─ modules/
│     ├─ constants.py            ← immutable lookup tables
│     ├─ predictors.py           ← YOLO, TT & doctr wrappers
│     └─ table_parser.py         ← converts TT boxes → markdown
doclayout_yolo_docstructbench_imgsz1024.pt   ← YOLO weights I’ve been experimenting with







