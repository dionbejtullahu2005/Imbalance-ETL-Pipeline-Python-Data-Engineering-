from src.extract import extract_excel
from src.transform import transform

df, summary, prices = extract_excel()

print("\nRAW DATA")
print(df.head())

print("\nROWS")
print(len(df))

print("\nCOLUMNS")
print(df.columns.tolist())

df = transform(df)

print("\nTRANSFORMED DATA")
print(
    df[
        [
            "datetime",
            "consumption_mwh",
            "production_mwh",
            "plan_mwh",
            "plan_mwh3",
            "imbalance",
            "imbalance_calculated",
            "price",
            "total_euro",
            "total_euro_calculated"
        ]
    ].head(10)
)

print("\nIMBALANCE DIFFERENCE")

difference = (
    df["imbalance"]
    -
    df["imbalance_calculated"]
)

print(
    difference.abs().max()
)

print("\nTOTAL EURO DIFFERENCE")

difference_euro = (
    df["total_euro"]
    -
    df["total_euro_calculated"]
)

print(
    difference_euro.abs().max()
)

print("\nPRICES")

print(
    prices.head()
)

print("\nSUMMARY")

print(
    summary.head()
)