import io
import sys
import time

import pandas as pd
import requests
import yfinance as yf

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCLUDE = r"Warrant|\bUnits?\b|\bRights?\b|Preferred|Notes due|Debenture|Depositary Shares, each"
MIN_TICKERS = 1000
BATCH = 200
PERIOD = "1y"


def parse_symbol_file(text, sym_col):
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df = df[~df[sym_col].fillna("").str.startswith("File Creation Time")]
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
    df = df[~df["Security Name"].str.contains(EXCLUDE, case=False, regex=True, na=False)]
    return df[sym_col].str.strip()


def load_symbols():
    parts = []
    for url, col in [(NASDAQ_URL, "Symbol"), (OTHER_URL, "ACT Symbol")]:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        parts.append(parse_symbol_file(resp.text, col))
    syms = pd.concat(parts)
    syms = syms[~syms.str.contains(r"[\$\^=]", regex=True)]
    return sorted(set(syms.str.replace(".", "-", regex=False)))


def to_long(df, tickers):
    frames = []
    have = set(df.columns.get_level_values(0))
    for t in tickers:
        if t not in have:
            continue
        sub = df[t].dropna(how="all")
        if sub.empty:
            continue
        sub = sub.reset_index()
        sub.columns = [str(c).lower() for c in sub.columns]
        sub["ticker"] = t
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download(symbols):
    out = []
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=PERIOD, interval="1d", auto_adjust=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print(f"batch {i}: attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(15 * (attempt + 1))
        if df is None or df.empty:
            continue
        long = to_long(df, chunk)
        if not long.empty:
            out.append(long)
        print(f"{min(i + BATCH, len(symbols))}/{len(symbols)} symbols processed")
        time.sleep(2)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(path):
    symbols = load_symbols()
    print(f"{len(symbols)} symbols in universe")
    data = download(symbols)
    if data.empty:
        sys.exit("No data downloaded")
    data = data.dropna(subset=["close"])
    data = data[["date", "ticker", "open", "high", "low", "close", "volume"]]
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    for c in ["open", "high", "low", "close"]:
        data[c] = data[c].astype("float32")
    data["volume"] = data["volume"].astype("float64")
    n = data["ticker"].nunique()
    print(f"{len(data)} rows, {n} tickers, last date {data['date'].max().date()}")
    if n < MIN_TICKERS:
        sys.exit(f"Only {n} tickers downloaded; not publishing")
    data.sort_values(["ticker", "date"]).to_parquet(path, index=False, compression="zstd")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "us_daily.parquet")
