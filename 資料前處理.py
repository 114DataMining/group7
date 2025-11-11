import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 1. 資料載入
print("1. 資料載入")
file_path = 'covid_vaccination_vs_death_ratio.csv'
df = pd.read_csv(file_path)
print(f"原始資料形狀: {df.shape}")

# 2. 缺失值處理
print("\n2. 缺失值處理")
missing_rate = df.isnull().mean() * 100
df = df.loc[:, missing_rate < 50] # 移除缺失率超過 50% 的欄位

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

# 數值使用中位數填補，類別使用 '未知' 填補
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna('未知')

print(f"缺失值已處理完畢，目前欄位數：{df.shape[1]}")

# 3. 移除異常值
print("\n3. 移除異常值")
before_rows = df.shape[0]

# 邏輯篩選：新死亡人數 >= 0 且 < 1百萬
if 'New_deaths' in df.columns:
    df = df[(df['New_deaths'] >= 0) & (df['New_deaths'] < 1_000_000)]
# 邏輯篩選：比例 (ratio) 在 0% 到 100% 之間
if 'ratio' in df.columns:
    df = df[(df['ratio'] >= 0) & (df['ratio'] <= 100)]

after_rows = df.shape[0]
print(f"已依邏輯規則移除異常值 {before_rows - after_rows} 筆，目前資料筆數: {after_rows}")

# 4. 新增資料
print("\n4. 新增資料")

# 4.1 建立洲別分類 (continent)
def country_to_continent(country_name):
    try:
        code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(code)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except:
        return "Unknown"

if 'country' in df.columns:
    df["continent"] = df["country"].apply(country_to_continent)
    # 手動修正常見的非標準國家名稱
    manual_fix = {
        "Taiwan": "Asia",
        "Hong Kong": "Asia",
        "Kosovo": "Europe",
        "Congo": "Africa",
        "South Sudan": "Africa",
        "Palestine": "Asia"
    }
    df["continent"] = df.apply(lambda x: manual_fix.get(x["country"], x["continent"]), axis=1)

    # 洲別增加檢查
    print("\n  >> 洲別分類結果檢查:")
    total_countries = df['country'].nunique()
    total_continents = df['continent'].nunique() 
    
    print(f"  資料集中包含 {total_countries} 個國家。") 
    print(f"  總共分類成 {total_continents} 個獨特洲別 (含 Unknown)。") 
    print("\n  各洲資料筆數:")
    print(df['continent'].value_counts().sort_index().to_string())

    # 4.2 建立接種率 (vaccination_rate)
if 'people_fully_vaccinated' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['people_fully_vaccinated'] / df['population'] * 100
elif 'people_vaccinated' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['people_vaccinated'] / df['population'] * 100
elif 'total_vaccinations' in df.columns and 'population' in df.columns:
    df['vaccination_rate'] = df['total_vaccinations'] / df['population'] * 100
else:
    print("找不到可計算接種率的欄位。")

# 限制接種率不超過 100%
if 'vaccination_rate' in df.columns:
    df.loc[df['vaccination_rate'] > 100, 'vaccination_rate'] = 100

# 清理欄位名稱中的前後空白
df.columns = df.columns.str.strip()


# 5. 新增各國各洲接種率及死亡率
print("\n5. 新增各國各洲接種率及死亡率")

# 各國平均
if 'country' in df.columns:
    country_avg = df.groupby('country')[['vaccination_rate', 'New_deaths']].transform('mean')
    df['country_avg_vaccination_rate'] = country_avg['vaccination_rate']
    df['country_avg_death_rate'] = country_avg['New_deaths']
else:
    print("Country 欄位不存在，略過各國平均計算。")

# 各洲平均
if 'continent' in df.columns:
    continent_avg = df.groupby('continent')[['vaccination_rate', 'New_deaths']].transform('mean')
    df['continent_avg_vaccination_rate'] = continent_avg['vaccination_rate']
    df['continent_avg_death_rate'] = continent_avg['New_deaths']
else:
    print("Continent 欄位不存在，略過各洲平均計算。")

print("新增各國與各洲平均接種率/死亡率欄位完成。")


# 6. 資料標準化
print("\n6. 資料標準化")

numeric_cols = [
    'vaccination_rate', 'New_deaths', 'ratio',
    'country_avg_vaccination_rate', 'country_avg_death_rate',
    'continent_avg_vaccination_rate', 'continent_avg_death_rate'
]

# 篩選實際存在的欄位
numeric_cols = [col for col in numeric_cols if col in df.columns]

scaler = StandardScaler()
# 確保 df 和 scaled_df 的索引一致
scaled_data = scaler.fit_transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_z" for col in numeric_cols])

# 合併標準化後的結果
df = pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
print("資料標準化 (Z-Score) 完成。")


# 7. 降維處理 (PCA)
print("\n7. 降維處理 (PCA)")

pca = PCA(n_components=2)
# PCA 在標準化後的數據上執行
pca_result = pca.fit_transform(scaled_df) 
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

print("主成分分析 (PCA) 完成。")
print("PCA 解釋變異比例:", pca.explained_variance_ratio_)


# 8. 輸出結果
print("\n8. 輸出結果")
output_path = 'covid_preprocessed.csv'
df.to_csv(output_path, index=False)
print(f"資料前處理腳本執行完畢，輸出檔案: {output_path}")

print("\n預覽（部分欄位）")
show_cols = [
    "country", "continent",
    "vaccination_rate", "death_rate_per_million",
    "country_avg_vaccination_rate", "country_avg_death_rate",
    "continent_avg_vaccination_rate", "continent_avg_death_rate"
]

