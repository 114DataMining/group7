import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# 2. 資料載入

print("1. 資料載入")
file_path = 'covid_vaccination_vs_death_ratio.csv'  # 剛剛上傳的檔名
df = pd.read_csv(file_path)
print(f"原始資料形狀: {df.shape}")


# 3. 缺失值處理

print("\n2. 缺失值處理")
missing_rate = df.isnull().mean() * 100
df = df.loc[:, missing_rate < 50]  # 移除缺失率 >50% 欄位

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna('未知')

print(f"缺失值已處理，目前欄位數：{df.shape[1]}")


# 4. 異常值移除

print("\n3. 移除異常值")
before = df.shape[0]

if 'New_deaths' in df.columns:
    df = df[(df['New_deaths'] >= 0) & (df['New_deaths'] < 1_000_000)]

if 'ratio' in df.columns:
    df = df[(df['ratio'] >= 0) & (df['ratio'] <= 100)]

after = df.shape[0]
print(f"移除異常值 {before - after} 筆")


# 5. 新增洲別欄位（使用 pycountry-convert）

print("\n4. 建立洲別")

def country_to_continent(country_name):
    try:
        code = pc.country_name_to_country_alpha2(country_name)
        cont = pc.country_alpha2_to_continent_code(code)
        return pc.convert_continent_code_to_continent_name(cont)
    except:
        return "Unknown"

if 'country' in df.columns:
    df["continent"] = df["country"].apply(country_to_continent)

    manual_fix = {
        "Taiwan": "Asia",
        "Hong Kong": "Asia",
        "Kosovo": "Europe",
        "Congo": "Africa",
        "South Sudan": "Africa",
        "Palestine": "Asia"
    }
    df["continent"] = df.apply(
        lambda x: manual_fix.get(x["country"], x["continent"]),
        axis=1
    )
else:
    print("⚠ 找不到 country 欄位，無法建立 continent，全部標為 Unknown")
    df["continent"] = "Unknown"


# 6. 新增接種率（百分比 %）

print("\n4.2 建立接種率 (百分比)")

if 'people_fully_vaccinated' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['people_fully_vaccinated'] / df['population'] * 100
elif 'people_vaccinated' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['people_vaccinated'] / df['population'] * 100
elif 'total_vaccinations' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['total_vaccinations'] / df['population'] * 100
else:
    print("⚠ 無法計算 vaccination_rate，缺少欄位。")

# 限制接種率不超過 100%
if 'vaccination_rate' in df.columns:
    df.loc[df['vaccination_rate'] > 100, 'vaccination_rate'] = 100


# 7. 新增死亡率（每百萬人口）

print("\n4.3 建立死亡率（每百萬人口）")

if 'New_deaths' in df.columns and 'population' in df.columns:
    df['death_rate_per_million'] = df['New_deaths'] / df['population'] * 1_000_000
else:
    print("⚠ 無法計算 death_rate_per_million，缺少欄位。")


# 8. 各國、各洲平均

print("\n5. 新增各國／各洲平均死亡率與接種率")

if 'country' in df.columns and 'vaccination_rate' in df.columns and 'death_rate_per_million' in df.columns:
    cavg = df.groupby('country')[['vaccination_rate','death_rate_per_million']].transform('mean')
    df['country_avg_vaccination_rate'] = cavg['vaccination_rate']
    df['country_avg_death_rate'] = cavg['death_rate_per_million']

if 'continent' in df.columns and 'vaccination_rate' in df.columns and 'death_rate_per_million' in df.columns:
    cav = df.groupby('continent')[['vaccination_rate','death_rate_per_million']].transform('mean')
    df['continent_avg_vaccination_rate'] = cav['vaccination_rate']
    df['continent_avg_death_rate'] = cav['death_rate_per_million']

print("各國/各洲平均欄位建立完成。")

# ============================
# 9. 數據標準化（Z-score）
# ============================
print("\n6. 資料標準化")

numeric_cols = [
    'vaccination_rate',
    'death_rate_per_million',
    'country_avg_vaccination_rate',
    'country_avg_death_rate',
    'continent_avg_vaccination_rate',
    'continent_avg_death_rate'
]

numeric_cols = [col for col in numeric_cols if col in df.columns]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_z" for col in numeric_cols])
df = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)
print("標準化完成！")


# 10. PCA 降維

print("\n7. PCA 降維")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
print("PCA 完成")
print("解釋變異比例:", pca.explained_variance_ratio_)


# 11. 輸出

df.to_csv("covid_preprocessed_final.csv", index=False)
print("\n covid_preprocessed_final.csv")
