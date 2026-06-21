# AI Sentiment and Crypto Volatility — Thesis Replication

This repository contains the reproducible code, data-processing scripts, and analysis used to reproduce the empirical results in the author's Bachelor thesis on AI-related sentiment and cryptocurrency volatility/connectedness.

Short abstract

This project studies how AI-related news and social-media discussion affect intraday and daily cryptocurrency volatility and connectedness. It implements a full replication workflow: data collection (GDELT, Reddit Pushshift, Kalshi, Metaculus, Yahoo Finance), RoBERTa-based sentiment scoring, time-series construction, EGARCH and ARMAX-EGARCHX volatility models, and TVP-VAR connectedness analyses. The repository contains both Python and R components to run the models used in the thesis and reproduce the paper tables and figures.

Repository highlights

- Languages: Python (main), R (modeling and TVP-VAR/EGARCH stages), Jupyter notebooks for EDA
- Reproducible workflow: scripts are organized into collection, cleaning, preparation, modeling, validation, and reporting stages
- Clear separation of raw, interim, and processed data (data/ is excluded from the repo; see Data Availability)
- MIT license

Quickstart (recommended)

1. Make sure you have Python 3.10+ and R installed. Use the Conda environment or venv below.

```bash
# Python (venv)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e code

# R
Rscript r/setup_renv.R
```

2. Place external raw files in data/raw/ as described in data/DATA_SOURCES_AND_PROCESSING.md.

3. Run the staged workflow from the repository root. Example (full workflow):

```bash
python scripts/collection/collect_yahoo_finance.py
python scripts/collection/extract_target_subreddits.py --input-folder data/raw/reddit/submissions --output-folder data/interim/reddit/target_subreddits
python scripts/cleaning/prepare_roberta_text_input.py
python scripts/modeling/run_roberta_sentiment.py
python scripts/preparation/build_combined_time_series.py
python scripts/reporting/make_thesis_tables.py
Rscript r/modeling/run_armax_egarchx.R
Rscript r/modeling/run_egarch_volatility.R
python scripts/modeling/run_tvpvar_connectedness_models.py
```

See the full workflow and per-stage commands in the original README sections and supporting documentation files.

Data availability and privacy

- This repository uses only public data sources. Large raw inputs are intentionally not tracked in Git (see .gitignore). The expected filenames and placement are documented in `data/DATA_SOURCES_AND_PROCESSING.md`.
- Before making the repository public, ensure you have removed any secret files, API keys, or credential material (search for .env, credentials.json, or hard-coded keys). Use `git rm --cached` for files that were accidentally committed and add them to .gitignore.

Suggested repository description (copy/paste for GitHub settings)

"Reproducible code and data workflow for a Bachelor thesis on AI-related sentiment and cryptocurrency volatility and connectedness (EGARCH, TVP-VAR)."

Suggested topics (copy/paste when editing topics)

econometrics, sentiment-analysis, egarch, tvp-var, volatility, crypto, cryptocurrency, thesis, reproducible-research, python, r, time-series, connectedness

How to make the repo public (web and CLI)

- Web UI: Settings → General → Change repository visibility → Make public. Follow the confirmation prompts.
- gh CLI (requires gh installed and authenticated):

```bash
gh repo edit BoazMuller/Bachelor-Thesis-AI-Sentiment-Crypto-Tech --visibility public
```

How to set description and topics via gh CLI or API

- Set description via gh CLI:

```bash
gh repo edit BoazMuller/Bachelor-Thesis-AI-Sentiment-Crypto-Tech --description "Reproducible code and data workflow for a Bachelor thesis on AI-related sentiment and cryptocurrency volatility and connectedness (EGARCH, TVP-VAR)."
```

- Set topics via GitHub API (use an access token with repo scope):

```bash
curl -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/BoazMuller/Bachelor-Thesis-AI-Sentiment-Crypto-Tech/topics \
  -d '{"names": ["econometrics","sentiment-analysis","egarch","tvp-var","volatility","crypto","cryptocurrency","thesis","reproducible-research","python","r","time-series","connectedness"]}'
```

Pinning the repository to your profile

1. Go to your GitHub profile page.
2. Click "Customize your profile" (or the pencil on the pinned repos area).
3. Add this repository to your pinned repositories.

Checklist before making the repo public

- [ ] Remove any secrets or credentials
- [ ] Ensure large data are not committed (use .gitignore and Git LFS if needed)
- [ ] Confirm license is correct (MIT included)
- [ ] Add or confirm citation for the thesis and author contact
- [ ] Add topics and a succinct description

Citation / How to cite

If you use this replication package, please cite: Boaz Muller (2026), "AI Sentiment and Cryptocurrency Volatility: [Thesis Title]." Contact: BoazMuller at GitHub (profile: https://github.com/BoazMuller).

License

This repository is released under the MIT License (see LICENSE).

Contact

Author: Boaz Muller — https://github.com/BoazMuller

---

(Edited to improve clarity and add a thesis-focused abstract, quickstart, and suggested description/topics for GitHub settings.)
