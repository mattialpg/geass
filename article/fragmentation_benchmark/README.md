# Migrated fragmentation benchmark results

These files were converted from the existing [`article/fragmentation_benchmark.csv`](../fragmentation_benchmark.csv) into the output layout of [`article/fragmentation_benchmark.py`](../fragmentation_benchmark.py).

**No fragmentation or molecular annotation was rerun.** The available CSV holds 900 evaluations (300 NP Atlas compounds × three methods). The notebook's configured five repetitions are not present in this source file, so the migrated `replicate=1` denotes the single recorded result for each compound/method.

## Files

- `runs.csv`: complete original per-method results, with a `replicate` field and the new script's columns. Unrecorded setup time, total time, heavy-atom count and molecular weight remain blank rather than estimated.
- `granularity.csv`: the same molecule-level rows (one observation per compound/method, including unsuccessful runs); this matches the script's `granularity.csv` semantics.
- `runtime_summary.csv`: timings and valid-output coverage using **all** recorded executions, including failed/no-output runs.
- `granularity_summary.csv`: fragment-count and size statistics calculated from **successful outputs only**, matching the revised Supplementary Table S3.
- `metadata.json`: input provenance and known limitations.

The source CSV does not contain individual fragment SMILES, so **`fragments.jsonl` cannot be reconstructed**. It is intentionally omitted; an empty file would incorrectly suggest the fragments were absent.

## Generate all figure exports, without rerunning the benchmark

From the repository root:

```bash
python article/fragmentation_benchmark.py \
  --from-csv article/fragmentation_benchmark/runs.csv \
  --output-dir article/fragmentation_benchmark
```

This command uses existing results only, refreshes the summary CSVs and writes `fig3a.svg/png`, `fig3b.svg/png` and `fig3.svg/png`. Image rendering requires the compatible Plotly/Kaleido dependencies described by the script. The figure files are not included in this migration until rendered.

**Scientific scope:** these imported data support the 300-compound evaluation and must not be presented as independently obtained 10,000-compound results.
