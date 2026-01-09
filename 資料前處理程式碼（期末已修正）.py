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


#   國家 → 洲別 

country_continent = {
    # Africa
    "AGO":"Africa","BEN":"Africa","BFA":"Africa","BWA":"Africa","CAF":"Africa",
    "CIV":"Africa","CMR":"Africa","COD":"Africa","COM":"Africa","CPV":"Africa",
    "DJI":"Africa","DZA":"Africa","EGY":"Africa","ERI":"Africa","ETH":"Africa",
    "GAB":"Africa","GHA":"Africa","GIN":"Africa","GMB":"Africa","GNB":"Africa",
    "GNQ":"Africa","KEN":"Africa","LBR":"Africa","LBY":"Africa","LSO":"Africa",
    "MAR":"Africa","MDG":"Africa","MLI":"Africa","MOZ":"Africa","MRT":"Africa",
    "MUS":"Africa","MWI":"Africa","NAM":"Africa","NER":"Africa","NGA":"Africa",
    "RWA":"Africa","SDN":"Africa","SEN":"Africa","SLE":"Africa","SOM":"Africa",
    "SSD":"Africa","STP":"Africa","TCD":"Africa","TGO":"Africa","TUN":"Africa",
    "TZA":"Africa","UGA":"Africa","ZAF":"Africa","ZMB":"Africa","ZWE":"Africa",
    "SYC":"Africa",
    
    # Asia
    "AFG":"Asia","ARM":"Asia","AZE":"Asia","BGD":"Asia","BRN":"Asia","BTN":"Asia",
    "CHN":"Asia","GEO":"Asia","IDN":"Asia","IND":"Asia","IRN":"Asia","IRQ":"Asia",
    "ISR":"Asia","JPN":"Asia","JOR":"Asia","KAZ":"Asia","KGZ":"Asia","KHM":"Asia",
    "KOR":"Asia","KWT":"Asia","LAO":"Asia","LBN":"Asia","LKA":"Asia","MDV":"Asia",
    "MMR":"Asia","MNG":"Asia","MYS":"Asia","NPL":"Asia","OMN":"Asia","PHL":"Asia",
    "PSE":"Asia","SGP":"Asia","SYR":"Asia","THA":"Asia","TJK":"Asia","TKM":"Asia",
    "TUR":"Asia","UZB":"Asia","VNM":"Asia","YEM":"Asia",

    # Europe
    "ALB":"Europe","AUT":"Europe","BEL":"Europe","BIH":"Europe","BGR":"Europe",
    "BLR":"Europe","CHE":"Europe","CYP":"Europe","CZE":"Europe","DEU":"Europe",
    "DNK":"Europe","ESP":"Europe","EST":"Europe","FIN":"Europe","FRA":"Europe",
    "GBR":"Europe","GRC":"Europe","HRV":"Europe","HUN":"Europe","IRL":"Europe",
    "ISL":"Europe","ITA":"Europe","LIE":"Europe","LTU":"Europe","LUX":"Europe",
    "LVA":"Europe","MDA":"Europe","MNE":"Europe","NLD":"Europe","NOR":"Europe",
    "POL":"Europe","PRT":"Europe","ROU":"Europe","SRB":"Europe","SVK":"Europe",
    "SVN":"Europe","SWE":"Europe","UKR":"Europe","MLT":"Europe","SMR":"Europe",
    "IMN":"Europe",

    # North America
    "CAN":"North America","CRI":"North America","CUB":"North America",
    "DMA":"North America","DOM":"North America","GTM":"North America",
    "HND":"North America","HTI":"North America","JAM":"North America",
    "KNA":"North America","MEX":"North America","PAN":"North America",
    "SLV":"North America","TTO":"North America","USA":"North America",
    "GRD":"North America","MSR":"North America",

    # South America
    "ARG":"South America","BOL":"South America","BRA":"South America",
    "CHL":"South America","COL":"South America","ECU":"South America",
    "GUY":"South America","PER":"South America","PRY":"South America",
    "SUR":"South America","URY":"South America","VEN":"South America",

    # Oceania
    "AUS":"Oceania","COK":"Oceania","FJI":"Oceania","NCL":"Oceania",
    "NZL":"Oceania","PNG":"Oceania","PYF":"Oceania","SLB":"Oceania",
    "VUT":"Oceania","WSM":"Oceania"
}


#   新增 Continent 欄位

df["Continent"] = df["iso_code"].map(country_continent)


#   檢查未分類國家

missing = df[df["Continent"].isna()]["iso_code"].unique()
if len(missing) > 0:
    print(" 以下國家未被分類：")
    print(missing)
else:
    print("🎉 所有國家皆成功分類")



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

#   刪除不必要的索引欄位

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("已刪除欄位：Unnamed: 0")


#   刪除關鍵欄位缺失的資料

key_columns = [
    "country", "date", "population",
    "total_vaccinations", "vaccination_rate"
]

missing_before = len(df)
df = df.dropna(subset=key_columns)
missing_after = len(df)


#   刪除不合理的數值資料
# 人口必須 > 0
pop_before = len(df)
df = df[df["population"] > 0]
pop_after = len(df)

# 疫苗接種率必須介於 0 和 1
rate_before = len(df)
df = df[(df["vaccination_rate"] >= 0) & (df["vaccination_rate"] <= 1)]
rate_after = len(df)

# 疫苗累積數不可為負
vac_before = len(df)
df = df[df["total_vaccinations"] >= 0]
vac_after = len(df)

print(f" 人口不合理刪除：{pop_before - pop_after} 筆")
print(f" 接種率不合理刪除：{rate_before - rate_after} 筆")
print(f" 疫苗數不合理刪除：{vac_before - vac_after} 筆")


#   刪除重複資料
#   （同一國家、同一天只留一筆）

dup_before = len(df)
df = df.drop_duplicates(subset=["country", "date"])
dup_after = len(df)

print(f" 重複資料刪除：{dup_before - dup_after} 筆")


# 6. 結果統計
before_count = len(df)
after_count = len(df)
removed_total = before_count - after_count

print("-" * 40)
print(" 清理後資料筆數 (After Cleaning):", after_count)
print(" 總共刪除筆數:", removed_total)
print("-" * 40)


# 7. 輸出清理後資料

output_file = "vaccination_rate_cleaned.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f" 清理完成，已輸出：{output_file}")