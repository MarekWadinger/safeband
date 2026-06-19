# Reproducing the I7 fault-typing paper

Every table, figure, and number in `main.pdf` is regenerated from the
committed code and data at the pinned git tag **`i7-fault-typing-v1`**.
Nothing is transcribed by hand.

```bash
REPO=safeband
git clone https://github.com/MarekWadinger/$REPO.git
cd $REPO
git checkout i7-fault-typing-v1
```

## 1. Environment (deterministic)

Python 3.12, with the exact dependency set pinned in `uv.lock`:

```bash
uv sync                       # recreates the locked environment
uv pip install --no-deps "TSB-AD==1.5" statsmodels   # benchmark-only deps (I6)
```

All benchmark scripts use a fixed seed (`RANDOM_STATE = 42`) and are
deterministic.

## 2. Data

Committed in the repo (no action needed):

- CATS nominal block — `examples/data/multivariate/cats/data_1t_agg_last.csv`
- SKAB — `examples/data/multivariate/alldata_skab.csv`

Fetched into `.temp/data/` from the original sources (not redistributed):

```bash
mkdir -p .temp/data/realfaults
# Intel Berkeley Lab (real battery-depletion faults)
curl -L https://db.csail.mit.edu/labdata/data.txt.gz \
     -o .temp/data/realfaults/intel_lab.txt.gz
# LBNL building-FDD (real outdoor-air-temp sensor bias) — OEDI submission 910
curl -L "https://data.openei.org/files/910/Data%20Sets%20for%20AFDD%20Evauluation%20of%20Building%20FDD%20Algorithms.zip" \
     -o .temp/data/realfaults/lbnl_fdd.zip
unzip -o .temp/data/realfaults/lbnl_fdd.zip -d .temp/data/realfaults/lbnl
# TSB-AD benchmark data + File_List from TheDatumOrg/TSB-AD into .temp/data/
```

DOIs: CATS `10.5281/zenodo.8338435`; LBNL `10.25984/1824861`;
UCI gas (not used in the paper) `10.24432/C5RP6W`.

## 3. Regenerate the result CSVs

Each script writes its CSV(s) under `examples/benchmarks/`:

```bash
uv run python examples/08_i7_scaled_validation.py      # per-type recall, bias-coupling
uv run python examples/09_i7_freeze_latency.py         # freeze vs CDF latency
uv run python examples/10_i7_regime_fp.py              # regime FP vs correlation
uv run python examples/11_i7_ablations.py              # dead-band, freeze-quiescent
uv run python examples/12_i7_intel_lab_realfaults.py   # Intel-Lab operating curve
uv run python examples/13_i7_lbnl_bias.py              # LBNL real-bias detection floor
uv run python examples/14_i5_reunanen_headtohead.py    # detection baseline
uv run python examples/06_tsb_ad_fullrun.py --split U     # TSB-AD (I6); univariate, resumable
```

`06_tsb_ad_fullrun.py` is the long one (multi-hour). It is fully
resumable: tuned hyperparameters cache to `.temp/tsb_ad_tuning/`, score
arrays to `.temp/tsb_ad_scores/`, and result rows append to the output
CSV per series — a crash loses at most the series in flight. The paper
reports the univariate split (`--split U`); the multivariate split
(`--split M`) is left to future work.

## 4. Render figures and build the PDF

```bash
uv run python publications/I7_fault_typing/make_figures.py   # 4 PDFs in figures/
cd publications/I7_fault_typing && latexmk -pdf main.tex
```

The figures are derived solely from the committed CSVs, so the rendered
`main.pdf` matches the tagged artifacts exactly.
