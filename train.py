import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def clean_and_train():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv("Data3.csv")

    # 2. Compute Price in Lakhs (consistent with the notebook)
    df['Price'] = (df['Total_Area(SQFT)'] * df['Price_per_SQFT'] / 10**5).round()

    # 3. Drop unnecessary columns and rename
    df = df.drop(['Unnamed: 0', 'Property_Name', 'Property Title', 'city', 'Description', 'BHK'], axis='columns')
    df = df.rename(columns={
        "Total_Area(SQFT)": "area",
        "Price_per_SQFT": "rate",
        "Total_Rooms": "BHK",
        "property_type": "type"
    })

    # 4. Standardize Location string to (locality, city)
    def clean_location(x):
        parts = x.split(',')
        if len(parts) >= 2:
            return parts[-2].strip() + ',' + parts[-1].strip()
        return x.strip()

    df.Location = df.Location.apply(clean_location)

    # 5. Filter area and price ranges
    df = df[(df.area >= 200) & (df.area <= 4000)]
    df = df[(df.Price < 1000) & (df.Price > 20)].copy()

    # 6. Encode Balcony as binary (1 for Yes, 0 for No)
    df.Balcony = df.Balcony.apply(lambda x: 1 if x == 'Yes' else 0)

    # 7. One-hot encode property type (Flat, House, Villa)
    type_dummies = pd.get_dummies(df['type'], prefix='type').astype(int)
    df = pd.concat([df.drop('type', axis='columns'), type_dummies], axis='columns')

    # 8. Group minor locations as 'other'
    location_stats = df.groupby('Location')['Location'].agg('count').sort_values(ascending=False)
    minor_locations = location_stats[location_stats <= 10]
    df['Location'] = df['Location'].apply(lambda x: 'other' if x in minor_locations else x)

    # 9. Outlier removal for BHK / area ratio (minimum 250 sqft per room)
    df = df[~(df['area'] / df.BHK < 250)]

    # 10. Outlier removal for price per SQFT (rate) per location within 1 standard deviation
    def remove_rate_outliers(data):
        df_out = pd.DataFrame()
        for key, subdf in data.groupby('Location'):
            m = np.mean(subdf['rate'])
            st = np.std(subdf['rate'])
            reduced_df = subdf[(subdf['rate'] > (m - st)) & (subdf['rate'] <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df = remove_rate_outliers(df)

    # 11. Drop 'other' locations and filter high rate listings
    df = df[~(df.Location == 'other')]
    df = df[df.rate <= 20000]

    # Drop the temporary rate column
    df = df.drop(['rate'], axis='columns')

    # 12. One-hot encode location columns
    location_dummies = pd.get_dummies(df['Location']).astype(int)
    df = pd.concat([df.drop('Location', axis='columns'), location_dummies], axis='columns')

    # 13. Split into features X and target y
    X = df.drop(['Price'], axis='columns')
    y = df.Price

    # 14. Train-Test Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    print(f"Features shape: {X.shape}")

    # 15. Train optimized XGBoost model
    print("Training optimized XGBoost model...")
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.2,
        max_depth=8,
        subsample=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate score
    score = model.score(X_test, y_test)
    print(f"Validation R2 Score: {score:.4f} ({score * 100:.2f}%)")

    # 16. Re-train on the entire dataset for production use
    print("Re-training model on entire dataset for production deployment...")
    final_model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.2,
        max_depth=8,
        subsample=1.0,
        random_state=42
    )
    final_model.fit(X, y)

    # 17. Save model and columns
    print("Saving model to Data3.pickle...")
    joblib.dump(final_model, 'Data3.pickle')

    print("Saving column metadata to columns.json...")
    columns = {
        'data columns': list(X.columns)
    }
    with open("columns.json", "w") as f:
        json.dump(columns, f)

    print("Training and serialization completed successfully!")

if __name__ == "__main__":
    clean_and_train()
