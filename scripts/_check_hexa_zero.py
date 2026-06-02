import pandas as pd

mc = pd.read_csv('data/main_characters.csv', encoding='utf-8-sig')
hf = pd.read_csv('data/hexa_fragments.csv', encoding='utf-8-sig',
                 usecols=['ocid','hexa_fragments_total','num_valid_months'])
df = mc.merge(hf, on='ocid', how='left')
total = len(df)

mask = (df['level'] >= 270) & (df['hexa_fragments_total'] == 0)
sub  = df[mask]
print(f'전체: {total}명')
print(f'270+ & hexa=0: {len(sub)}명 ({len(sub)/total*100:.1f}%)')
print()
print('level 분포:')
print(sub['level'].value_counts().sort_index().to_string())
print()
print('class_group 분포:')
print(sub['class_group'].value_counts().to_string())

fm = pd.read_csv('data/features_monthly.csv', encoding='utf-8-sig',
                 usecols=['ocid','avg_monthly_delta_level','recent6_delta_level'])
sub2 = sub.merge(fm, on='ocid', how='left')
print()
print('avg_monthly_delta_level 분포 (270+ & hexa=0):')
print(sub2['avg_monthly_delta_level'].describe().round(3).to_string())
parked = (sub2['avg_monthly_delta_level'] <= 0.1).mean() * 100
print(f'delta_level <= 0.1 비율: {parked:.1f}%')
