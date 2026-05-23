# 1. استيراد المكتبات | Import Libraries
# ============================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# mlxtend هي المكتبة التي تحتوي على تطبيق Apriori جاهز
# قم بتثبيتها إذا لم تكن موجودة: pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ============================================================
# 2. تحميل البيانات | Load Data
# ============================================================
# تأكد من وجود ملف adult.csv في نفس المجلد
df = pd.read_csv("adult.csv")

print("=" * 60)
print("معلومات البيانات الأساسية | Dataset Info")
print("=" * 60)
print(f"عدد الصفوف: {df.shape[0]}")
print(f"عدد الأعمدة: {df.shape[1]}")
print(f"\nأسماء الأعمدة:\n{list(df.columns)}")
print(f"\nعينة من البيانات:")
print(df.head(3))

# ============================================================
# 3. تنظيف البيانات | Data Cleaning
# ============================================================
# استبدال القيم المفقودة "?" بـ NaN ثم حذفها
df = df.replace("?", np.nan)
df = df.dropna()
df = df.drop_duplicates()

print(f"\nحجم البيانات بعد التنظيف: {df.shape}")

# ============================================================
# 4. تحضير البيانات لخوارزمية Apriori
#    Data Preparation for Apriori
# ============================================================
"""
خوارزمية Apriori تعمل على بيانات فئوية (Categorical) فقط.
لذلك سنقوم بـ:
  أ. تحويل الأعمدة الرقمية إلى فئات (تقسيم إلى نطاقات)
  ب. دمج اسم العمود مع القيمة لتكوين "عنصر" واضح
     مثال: "age=Young" أو "target=>50K"
"""

# --- أ. تقسيم العمر إلى فئات | Bin Age ---
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 45, 65, 100],
    labels=['age=Young(≤25)', 'age=Middle(26-45)', 'age=Senior(46-65)', 'age=Elder(>65)']
)

# --- ب. تقسيم ساعات العمل | Bin Hours per Week ---
df['hours_group'] = pd.cut(
    df['hours.per.week'],
    bins=[0, 35, 45, 100],
    labels=['hours=PartTime(≤35)', 'hours=FullTime(36-45)', 'hours=OverTime(>45)']
)

# --- ج. اختيار الأعمدة المناسبة للتحليل ---
# نختار أعمدة فئوية ذات معنى + العمود الهدف
selected_cols = [
    'age_group',          # الفئة العمرية
    'workclass',          # نوع العمل
    'education',          # مستوى التعليم
    'marital.status',     # الحالة الاجتماعية
    'occupation',         # المهنة
    'relationship',       # العلاقة الأسرية
    'race',               # العرق
    'sex',                # الجنس
    'hours_group',        # ساعات العمل
    'target'              # العمود الهدف (الدخل)
]

df_apriori = df[selected_cols].copy()

# --- د. إضافة اسم العمود كبادئة لكل قيمة (لتمييز العناصر) ---
for col in df_apriori.columns:
    if col not in ['age_group', 'hours_group']:  # هذه تحتوي على البادئة بالفعل
        df_apriori[col] = col + "=" + df_apriori[col].astype(str)

print("\nعينة من البيانات المحضّرة للخوارزمية:")
print(df_apriori.head(3))

# ============================================================
# 5. تحويل البيانات إلى صيغة One-Hot (صح/خطأ لكل عنصر)
#    Convert to One-Hot Encoded Transactions
# ============================================================
"""
Apriori تحتاج كل صف أن يكون قائمة من العناصر الموجودة.
TransactionEncoder يحوّل القوائم إلى مصفوفة True/False.
"""

# تحويل كل صف إلى قائمة من العناصر
transactions = df_apriori.values.tolist()

# ترميز المعاملات
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print(f"\nشكل مصفوفة الترميز: {df_encoded.shape}")
print("عينة من البيانات المرمّزة:")
print(df_encoded.iloc[:3, :5])

# ============================================================
# 6. تطبيق خوارزمية Apriori
#    Apply Apriori Algorithm
# ============================================================
"""
min_support: الحد الأدنى للدعم (نسبة ظهور العنصر في البيانات)
نختار 0.1 (10%) لنحصل على نتائج معقولة دون إغراق بالمعلومات.
"""

print("\n" + "=" * 60)
print("تطبيق خوارزمية Apriori...")
print("=" * 60)

MIN_SUPPORT = 0.10      # الحد الأدنى للدعم (10%)
MIN_CONFIDENCE = 0.60   # الحد الأدنى للثقة (60%)
MIN_LIFT = 1.2          # الحد الأدنى للرفع

# استخراج مجموعات العناصر المتكررة (Frequent Itemsets)
frequent_itemsets = apriori(
    df_encoded,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=4            # الحد الأقصى لحجم المجموعة
)

# ترتيب النتائج تنازلياً حسب الدعم
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(
    lambda x: ", ".join(sorted(list(x)))
)

print(f"\nعدد مجموعات العناصر المتكررة: {len(frequent_itemsets)}")
print("\nأعلى 10 مجموعات من حيث الدعم:")
print(frequent_itemsets[['support', 'itemsets_str']].head(10).to_string(index=False))

# ============================================================
# 7. استخراج قواعد الارتباط | Extract Association Rules
# ============================================================
"""
association_rules تستخرج العلاقات بين العناصر بالشكل:
  إذا (antecedents) → إذن (consequents)

نحن مهتمون بالقواعد التي تتنبأ بالعمود الهدف (target)
"""

# استخراج جميع القواعد
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

# تصفية: نريد فقط القواعد التي نتيجتها تتعلق بـ target
rules_target = rules[
    rules['consequents'].apply(
        lambda x: any('target=' in item for item in x)
    )
].copy()

# تصفية إضافية بناءً على Lift
rules_target = rules_target[rules_target['lift'] >= MIN_LIFT]

# ترتيب حسب الثقة ثم الرفع
rules_target = rules_target.sort_values(
    ['confidence', 'lift'], ascending=False
).reset_index(drop=True)

# تحويل frozensets إلى نصوص مقروءة
rules_target['antecedents_str'] = rules_target['antecedents'].apply(
    lambda x: " AND ".join(sorted(list(x)))
)
rules_target['consequents_str'] = rules_target['consequents'].apply(
    lambda x: " AND ".join(sorted(list(x)))
)

print("\n" + "=" * 60)
print("قواعد الارتباط المرتبطة بالعمود الهدف (target)")
print(f"(min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT})")
print("=" * 60)
print(f"عدد القواعد المستخرجة: {len(rules_target)}")

# ============================================================
# 8. عرض أهم النتائج | Display Top Results
# ============================================================

display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']

print("\n--- أعلى 15 قاعدة ارتباط ---")
top_rules = rules_target[display_cols].head(15)
for i, row in top_rules.iterrows():
    print(f"\n[{i+1}] إذا: {row['antecedents_str']}")
    print(f"     إذن: {row['consequents_str']}")
    print(f"     الدعم={row['support']:.3f} | الثقة={row['confidence']:.3f} | الرفع={row['lift']:.3f}")

# ============================================================
# 9. تقسيم النتائج حسب الهدف | Split by Target
# ============================================================

rules_high = rules_target[
    rules_target['consequents_str'].str.contains('>50K')
].head(10)

rules_low = rules_target[
    rules_target['consequents_str'].str.contains('<=50K')
].head(10)

print("\n" + "=" * 60)
print("قواعد تشير إلى الدخل المرتفع (>50K) - أقوى 10 قواعد")
print("=" * 60)
for i, row in rules_high.iterrows():
    print(f"  [{i+1}] {row['antecedents_str']}")
    print(f"       → {row['consequents_str']}")
    print(f"       ثقة={row['confidence']:.2%} | رفع={row['lift']:.2f}")

print("\n" + "=" * 60)
print("قواعد تشير إلى الدخل المنخفض (<=50K) - أقوى 10 قواعد")
print("=" * 60)
for i, row in rules_low.iterrows():
    print(f"  [{i+1}] {row['antecedents_str']}")
    print(f"       → {row['consequents_str']}")
    print(f"       ثقة={row['confidence']:.2%} | رفع={row['lift']:.2f}")

# ============================================================
# 10. حفظ النتائج | Save Results
# ============================================================
rules_target[display_cols].to_csv("apriori_rules_output.csv", index=False, encoding='utf-8-sig')
frequent_itemsets[['support', 'itemsets_str']].to_csv("frequent_itemsets_output.csv", index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("تم حفظ النتائج في:")
print("  - apriori_rules_output.csv    (قواعد الارتباط)")
print("  - frequent_itemsets_output.csv (مجموعات العناصر المتكررة)")
print("=" * 60)

# ============================================================
# 11. ملخص الإحصائيات | Summary Statistics
# ============================================================
print("\n--- ملخص إحصائيات القواعد ---")
print(f"متوسط الثقة:  {rules_target['confidence'].mean():.3f}")
print(f"أعلى ثقة:    {rules_target['confidence'].max():.3f}")
print(f"متوسط الرفع: {rules_target['lift'].mean():.3f}")
print(f"أعلى رفع:    {rules_target['lift'].max():.3f}")
