# Obtaining FI-2010 (do not scrape)

FI-2010 is the scientific dataset for this project (Ntakaris et al., 2018,
Journal of Forecasting). It is **not** redistributed here.

## Official route

1. Read the paper: Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis,
   "Benchmark dataset for mid-price forecasting of limit order book data
   with machine learning methods", Journal of Forecasting, 2018.
2. Dataset page (ETS / Tampere / Helsinki authors): search
   "FI-2010 limit order book dataset Ntakaris".
   A commonly cited landing page is maintained by the original authors
   (Aalto / Tampere / Aristotle affiliates). Follow **their** licence
   and citation requirements.
3. Place the extracted files under `data/raw/FI-2010/` so that
   `src/data/fi2010.py` can find them. Expected layout is documented in
   the module docstring (40 features, 10 days, 5 stocks).

## What this repo will not do

- No automated download script that hits a private or ToS-gated host.
- No scraping of Nasdaq Nordic or vendor feeds.
- If `data/raw/FI-2010/` is missing, the code uses a **synthetic TOY**
  generator. Toy numbers must never be reported as FI-2010 results.

## Citation

Ntakaris et al. (2018). Please also cite Zhang, Zohren, Roberts (2019)
arXiv:1808.03668 when discussing DeepLOB, Wallbridge (2020) arXiv:2003.00130
for TransLOB, and the 2024 survey arXiv:2403.09267 for later LOB-ML context.
