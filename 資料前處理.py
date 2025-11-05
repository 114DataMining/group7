# 資料前處理.py

import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#1. 匯入資料
file_path = 'covid_vaccination_vs_death_ratio.csv'
df = pd.read_csv(file_path)
print(" Step 1：原始資料形狀 =", df.shape)

#2. 缺失值處理
missing_rate = df.isnull().mean() * 100
df = df.loc[:, missing_rate < 50]

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna('未知')

print(f"缺失值已處理完畢，目前欄位數：{df.shape[1]}")

#異常值處理 (IQR 夾回法)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

print("異常值已處理完成")

#新增衍生欄位
# 建立接種率
if 'people_fully_vaccinated' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['people_fully_vaccinated'] / df['population'] * 100
elif 'people_vaccinated' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['people_vaccinated'] / df['population'] * 100
elif 'total_vaccinations' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['total_vaccinations'] / df['population'] * 100
else:
    print(" 找不到可計算接種率的欄位，請確認 CSV 欄名！")

df.loc[df['vaccination_rate'] > 100, 'vaccination_rate'] = 100

# 洲別分類
def country_to_continent(country_name):
    try:
        code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(code)
        return pc.convert_continent_code_to_continent_name(continent_code)
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
    df["continent"] = df.apply(lambda x: manual_fix.get(x["country"], x["continent"]), axis=1)

# 欄位名稱去除空格
df.columns = df.columns.str.strip()

# 新增死亡率與完整接種率
if 'New_deaths' in df.columns and 'population' in df.columns:
    df["death_rate_per_million"] = (df["New_deaths"] / df["population"]) * 1_000_000

if 'people_fully_vaccinated' in df.columns and 'population' in df.columns:
    df["fully_vaccinated_rate"] = (df["people_fully_vaccinated"] / df["population"]) * 100

#5. 特徵選擇與標準化
features = ["vaccination_rate", "fully_vaccinated_rate", "death_rate_per_million", "ratio"]
df = df.dropna(subset=features)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_z" for col in features])
df = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

print("特徵標準化完成")

#6. PCA 降維
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df["PC1"] = pca_data[:, 0]
df["PC2"] = pca_data[:, 1]

print("PCA 完成（已新增欄位 PC1、PC2）")
print("主成分解釋變異比例：", pca.explained_variance_ratio_)

#7. 匯出最終結果
df.to_csv("step8_pca_result.csv", index=False)
print("已輸出最終檔案：step8_pca_result.csv")

