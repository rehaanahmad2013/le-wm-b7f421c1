# Paper-to-Marimo

The interactive companion is `paper-to-marimo.py`. Run it without modifying
your environment:

```bash
uv run --with marimo --with pandas --with altair marimo edit paper-to-marimo.py
```

Use `marimo run paper-to-marimo.py` for a read-only app. The notebook embeds the
paper targets and reproduced ORX metrics, keeps pending measurements explicit,
and includes the protocol discrepancies required to interpret the comparison.
