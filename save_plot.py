from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

xlsx = Path("demand.xlsx")
df_data = pd.read_excel(xlsx, sheet_name="Data").set_index("Make")
df_fc   = pd.read_excel(xlsx, sheet_name="Forecast").set_index("Make")

# Pick a Make with enough history
make = "Toyota"  # replace if Toyota isn't in your dataset
hist = df_data.drop(columns=["Total_Units"], errors="ignore").loc[make].dropna()
hist.index = pd.to_datetime(hist.index, format="%Y-%m")

fc = df_fc.loc[make]
fc.index = pd.to_datetime(fc.index, format="%Y-%m")

plt.figure(figsize=(9, 4.8))
plt.plot(hist.tail(24), label="History (last 24m)")
plt.plot(fc, label="Forecast (next 6m)", linestyle="--", marker="o")
plt.title(f"{make} — Demand Forecast")
plt.xlabel("Month")
plt.ylabel("Units")
plt.legend()
plt.tight_layout()
Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/forecast_plot.png", dpi=180)
print("✅ Saved docs/forecast_plot.png")