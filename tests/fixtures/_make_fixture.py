"""Regenerate tests/fixtures/sample_raw.csv from the REAL raw file.

Run this once against your S3 raw object so the CI test job runs offline on
representative data instead of the synthetic stand-in:

    S3_BUCKET=<bucket> RAW_KEY='<your raw key>' python tests/fixtures/_make_fixture.py

It writes a 600-row stratified sample. No credentials are stored — it uses the
same s3_io credential chain the app uses.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import s3_io  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "sample_raw.csv")

df = s3_io.read_csv_s3(os.environ["RAW_KEY"])
df.sample(n=min(600, len(df)), random_state=42).to_csv(OUT, index=False)
print(f"wrote {OUT} rows={min(600, len(df))} cols={len(df.columns)}")
