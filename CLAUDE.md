# CurrencyCapPortal — static price fallback for Gheymat

This repo is a single script plus a GitHub Actions workflow, and it is the Gheymat app's only data path: every
six minutes it fetches its sources, combines them, and publishes a sealed static JSON API on GitHub Pages that
the app polls. The old Flask API, ML predictions, auth and Railway deployment were removed on 2026-10-14; git
history keeps them.

Sources (Gheymat `CONTRACT.md` "Static publisher sources", the authority): `GHEYMAT_SOURCES` (default
`hansha,bonbast`) is a priority list; every source is fetched, each code takes the first source's valid value,
a failed or stale (>30 min) source is skipped, and `fetchedAt` is the oldest observation used so the app's
freshness pill stays honest. hansha has no `ounce` and prices `btc` in toman, so those two always come from
bonbast; history, long history and the computed cross-rates stay bonbast/archive only.

```
tools/gheymat_snapshot.py      sources → dist/v1/latest, dist/v1/instruments, dist/health,
                               dist/v1/daily.json + dist/v1/history/<code> (archive backfill, 12 days every run), dist/v1/longhistory/<code>, dist/v1/intraday/<code>, dist/index.html
tools/instruments.json         51 instruments, the source of truth for the static channel: 37 direct (bonbast key
                               set) + 6 hansha-only toman crypto rows (`bonbast_key: null`)
                               + 8 computed cross-rate currencies (`computed: true`). The exact formula, pivot-integrity check and
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
