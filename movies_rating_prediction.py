import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv(
    r"C:\titanicccc'\IMDb Movies India.csv",
    encoding="latin1"
)

print(df.head())

df = df.dropna(subset=["Rating"])

df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Unknown")
df["Actor 1"] = df["Actor 1"].fillna("Unknown")
df["Actor 2"] = df["Actor 2"].fillna("Unknown")
df["Actor 3"] = df["Actor 3"].fillna("Unknown")

df["Duration"] = df["Duration"].astype(str).str.replace(" min", "", regex=False)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
df["Duration"] = df["Duration"].fillna(df["Duration"].median())

df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
df["Votes"] = df["Votes"].fillna(df["Votes"].median())

df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Year"] = df["Year"].fillna(df["Year"].median())

genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor1_encoder = LabelEncoder()
actor2_encoder = LabelEncoder()
actor3_encoder = LabelEncoder()

df["Genre"] = genre_encoder.fit_transform(df["Genre"])
df["Director"] = director_encoder.fit_transform(df["Director"])
df["Actor 1"] = actor1_encoder.fit_transform(df["Actor 1"])
df["Actor 2"] = actor2_encoder.fit_transform(df["Actor 2"])
df["Actor 3"] = actor3_encoder.fit_transform(df["Actor 3"])

X = df[
    [
        "Year",
        "Duration",
        "Votes",
        "Genre",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3"
    ]
]

y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

sample_movie = X_test.iloc[[0]]

predicted_rating = model.predict(sample_movie)

print("\nPredicted Rating:", round(predicted_rating[0], 2))
print("Actual Rating:", y_test.iloc[0])