import argparse, calendar, requests, pandas as pd
from bs4 import BeautifulSoup
from datetime import date

HDRS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def month_url(year:int, month:int)->str:
    return f"https://wdwpassport.com/past-crowds/{calendar.month_name[month].lower()}-{year}"

def fetch(url):
    r = requests.get(url, headers=HDRS, timeout=20); r.raise_for_status(); return r.text

def parse_month(html, year, month):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    # Heuristic: look for daily tiles; fallback to parsing any "Crowds x/10" text per day
    for tile in soup.find_all(True, attrs={"class": lambda c: c and ("day" in c or "calendar" in c)}):
        txt = tile.get_text(" ", strip=True)
        crowd = None
        for tok in txt.replace("/", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= 10:
                crowd = int(tok); break
        if crowd is None:
            continue
        # infer day number
        dnum = None
        for t in txt.split():
            if t.isdigit():
                dn = int(t)
                try:
                    rows.append({"date": date(year, month, dn).isoformat(), "crowd_index": crowd})
                    break
                except Exception:
                    pass
    if not rows:
        raise RuntimeError("No days parsed; page layout may have changed.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    url = month_url(args.year, args.month)
    html = fetch(url)
    df = parse_month(html, args.year, args.month)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows")

if __name__ == "__main__":
    main()
