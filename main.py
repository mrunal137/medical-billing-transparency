import pandas as pd
from src.analysis import run_ttest, run_anova, run_correlation
from src.visualizations import plot_distribution, plot_bar, plot_box
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("data/hospital_billing_dataset.csv")

# =====================
# 📊 EDA
# =====================
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# Plots
plot_distribution(df['final_bill'], "Final Bill Distribution")
plot_bar(df, 'city', 'final_bill', "Average Final Bill by City")
plot_box(df, 'procedure', 'final_bill', "Final Bill by Procedure")

# =====================
# 🧪 Hypothesis Testing
# =====================
delhi = df[df['city'] == 'Delhi']['final_bill']
mumbai = df[df['city'] == 'Mumbai']['final_bill']
t_stat, p_val = run_ttest(delhi, mumbai)
print(f"\nT-Test (Delhi vs Mumbai):\nT-statistic = {t_stat:.2f}, P-value = {p_val:.4f}")

groups = [group['final_bill'].values for name, group in df.groupby('city')]
f_stat, p_anova = run_anova(groups)
print(f"\nANOVA (Final Bill by City):\nF-statistic = {f_stat:.2f}, P-value = {p_anova:.4f}")

corr, p_corr = run_correlation(df['insurance_discount'], df['final_bill'])
print(f"\nCorrelation (Insurance Discount vs Final Bill):\nr = {corr:.2f}, P-value = {p_corr:.4f}")

# =====================
# 🤖 ML Model
# =====================
features = df[['cost', 'room_charge', 'lab_fee', 'insurance_discount']]
target = df['final_bill']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nML Model Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
