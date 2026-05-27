# =========================
# 1. LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("BankChurners.csv")

# Kaggle-dan gələn lazımsız son sütunları təmizləyirik (əgər varsa xəta verməsin)
naive_bayes_cols = [c for c in df.columns if "Naive_Bayes" in c]
if naive_bayes_cols:
    df.drop(columns=naive_bayes_cols, inplace=True)

# =========================
# 3. CLEANING
# =========================
# Drop ID column (not useful for ML)
df.drop(["CLIENTNUM"], axis=1, inplace=True, errors='ignore')

# Target encoding (SAFE VERSION)
df["Attrition_Flag"] = df["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

# =========================
# 4. BASIC INFO CHECK
# =========================
print(df.info())
print(df.isnull().sum())

# =========================
# 5. EDA (BASIC INSIGHTS)
# =========================

print("Churn Rate:", df["Attrition_Flag"].mean())

df["Attrition_Flag"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.show()

plt.hist(df[df["Attrition_Flag"] == 1]["Customer_Age"], bins=20)
plt.title("Churn by Age")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 6. BUSINESS CX INSIGHTS
# =========================

print("\nChurn by Gender:")
print(df.groupby("Gender")["Attrition_Flag"].mean())

print("\nChurn by Income:")
print(df.groupby("Income_Category")["Attrition_Flag"].mean())

print("\nChurn by Card Type:")
print(df.groupby("Card_Category")["Attrition_Flag"].mean())

# =========================
# 7. CX FEATURE ENGINEERING
# =========================

# Engagement score
df["Engagement"] = df["Total_Trans_Ct"] / df["Months_on_book"]

# Active indicator
df["Active_Indicator"] = df["Total_Trans_Ct"].apply(lambda x: 1 if x > 50 else 0)

# Low balance flag
df["Low_Balance"] = df["Total_Revolving_Bal"].apply(lambda x: 1 if x < 500 else 0)

# Tenure groups
df["Tenure_Group"] = pd.cut(
    df["Months_on_book"],
    bins=[0, 24, 48, 100],
    labels=["New", "Mid", "Long"]
)

# =========================
# 8. CX SEGMENT INSIGHTS
# =========================

print("\nChurn by Active Indicator:")
print(df.groupby("Active_Indicator")["Attrition_Flag"].mean())

print("\nChurn by Low Balance:")
print(df.groupby("Low_Balance")["Attrition_Flag"].mean())

print("\nChurn by Tenure Group:")
print(df.groupby("Tenure_Group")["Attrition_Flag"].mean())


# =========================
# 🔥 DÜZƏLİŞ EDİLMİŞ HİSSƏ: 9. DATA PREPARATION (ML)
# =========================

# SƏHVİN HƏLLİ 1: Power BI üçün təmiz, True/False olmayan, kateqoriyalı datanı eksport edirik
df.to_csv("bank_churn_final.csv", index=False)

# SƏHVİN HƏLLİ 2: ML modelinin mətn sütunlarında çökməməsi üçün One-Hot Encoding edirik
categorical_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", "Tenure_Group"]
df_ml = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# İndi Feature (X) və Target (y) ayıra bilərik
X = df_ml.drop("Attrition_Flag", axis=1)
y = df_ml["Attrition_Flag"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (İndi bütün sütunlar rəqəm olduğu üçün StandardScaler problemsiz işləyəcək)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 10. MODEL TRAINING
# =========================

model = LogisticRegression(max_iter=1000)
# DÜZƏLİŞ: Model scaled olunmuş data ilə öyrənməlidir
model.fit(X_train_scaled, y_train)

# DÜZƏLİŞ: Proqnoz scaled test datası üzərindən aparılmalıdır
y_pred = model.predict(X_test_scaled)

# =========================
# 11. MODEL EVALUATION
# =========================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 12. FEATURE IMPORTANCE (INTERVIEW GOLD)
# =========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nTop Features Driving Churn:")
print(importance.head(10))

importance.head(10).plot(kind="bar", x="Feature", y="Importance")
plt.title("Top Churn Drivers")
plt.show()

# Pro Power BI qrafiki üçün bu əmsalları ayrıca fayl kimi çıxarırıq
importance.to_csv("churn_drivers_importance.csv", index=False)