# Assets

Static assets for the KAMUI documentation site and README.

- `DIAGRAMS.md` — all Mermaid architecture diagrams (source of truth)
- `logo.svg` — KAMUI logo (to be created)

## Generating diagrams as images

Install mermaid-cli:
```bash
npm install -g @mermaid-js/mermaid-cli
```

Render all diagrams:
```bash
mmdc -i assets/DIAGRAMS.md -o assets/ --outputFormat svg
```
