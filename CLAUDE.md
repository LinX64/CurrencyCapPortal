# CurrencyCapPortal — static price fallback for Gheymat

This repo is now a single script plus a GitHub Actions workflow. It fetches bonbast.com every six minutes
and publishes a static JSON API on GitHub Pages that the Gheymat iOS app can poll when the real server
(`/Users/mohsen/StudioProjects/Gheymat/server`) is not deployed. The old Flask API, ML predictions,
auth and Railway deployment were removed on 2026-10-14; git history keeps them.

```
tools/gheymat_snapshot.py      bonbast protocol → dist/v1/latest, dist/v1/instruments, dist/health,
                               dist/v1/daily.json + dist/v1/history/<code> (archive backfill, 12 days every run), dist/v1/longhistory/<code>, dist/v1/intraday/<code>, dist/index.html
tools/instruments.json         45 instruments: 37 direct (copied from Gheymat/server/gheymat_server/instruments.json)
                               + 8 computed cross-rate currencies (`computed: true`), same 8 rows as that
                               file's own computed entries. The exact formula, pivot-integrity check and
                               failure handling are specified once, authoritatively, in Gheymat's
                               `CONTRACT.md` ("Computed cross-rates") - this repo's `fetch_converter` /
                               `compute_cross_rates` / `fetch_computed_rates` in gheymat_snapshot.py
                               implement that spec as a stateless-per-run adaptation (no persisted
                               converter-table cache, unlike the server's `ConverterStore`).
.github/workflows/gheymat-latest.yml   hourly cron backstop (`17 * * * *`) that loops internally, publishing
                               dist/ to gh-pages every ~6 minutes for ~5.5 hours per run
archive/                       historical series kept as seed data (all.json back to 1978, 1y.json)
tests/                         pytest for the script (no network)
```

Live: https://linx64.github.io/CurrencyCapPortal/v1/latest (ETag, 304 on If-None-Match).

Rules: no new runtime dependencies beyond `requests`; the wire shapes follow Gheymat/CONTRACT.md exactly;
the workflow must stay under GitHub Pages' 10 builds/hour; never reintroduce the bonbast-history writer
that duplicated payloads into the archive (sustained history calls get yesterday's payload with HTTP 200).

Run locally: `pip install -r requirements.txt && python tools/gheymat_snapshot.py --out dist --history && pytest -q`.
