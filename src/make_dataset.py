# src/make_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    # Features
    tenure_months = rng.integers(1, 72, size=n)                # antigüedad
    monthly_charges = rng.normal(65, 20, size=n).clip(10, 200) # $
    contract_type = rng.choice(["month-to-month","one-year","two-year"], size=n, p=[0.6,0.25,0.15])
    payment_method = rng.choice(["credit-card","debit","transfer","paypal"], size=n)
    is_fiber = rng.choice([0,1], size=n, p=[0.45,0.55])
    promo_active = rng.choice([0,1], size=n, p=[0.3,0.7])
    support_tickets_90d = rng.poisson(1.2, size=n)
    app_logins_30d = rng.poisson(18, size=n)

    # Probabilidad de churn (modelo generativo)
    # intuición: contratos cortos, tickets altos, poco login, fibra cara, meses de tenure bajos -> más churn
    z = (
        -1.2
        + 0.015*(200 - monthly_charges) * -1   # cargos altos = mayor churn
        + 0.03*(2 - (tenure_months/36))        # tenure bajo = mayor churn
        + 0.35*is_fiber                        # fibra + costo => +churn
        + 0.28*support_tickets_90d             # +tickets => +churn
        - 0.02*app_logins_30d                  # +uso => -churn
        - 0.4*promo_active                     # promo reduce churn
        + np.select(
            [
                contract_type=="month-to-month",
                contract_type=="one-year",
                contract_type=="two-year",
            ],
            [0.6, 0.2, -0.2], 0.0
        )
    )
    p = sigmoid(z)
    churn = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "contract_type": contract_type,
        "payment_method": payment_method,
        "is_fiber": is_fiber,
        "promo_active": promo_active,
        "support_tickets_90d": support_tickets_90d,
        "app_logins_30d": app_logins_30d,
        "churn": churn
    })

    outdir = Path("data/raw"); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "churn_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out.resolve()}  rows={len(df)}  churn_rate={df['churn'].mean():.3f}")

if __name__ == "__main__":
    main()
