#!/usr/bin/env python3
"""
Корреляционный анализ набора данных вузов Москвы (data.mos.ru, набор 3326).

Запуск:
    python analyze_education.py data.xlsx
    python analyze_education.py data.csv --header-row 2
    python analyze_education.py data.xlsx --sheet Data --header-row 2

По умолчанию ожидается 3 строки-заголовка (англ., рус., технические имена),
поэтому header_row = 2 (индекс строки с техническими именами, 0-based).

Зависимости:
    pip install pandas numpy scipy openpyxl
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ============================ 1. Загрузка ============================

FIELD_ALIASES = {
    'name':           ['Full name', 'name', 'Полное официальное наименование организации'],
    'admin_area':     ['Administrative area', 'admin_area', 'Административный округ'],
    'phone':          ['Public phone', 'phone', 'Контактный телефон'],
    'email':          ['E-mail', 'email', 'Email', 'Адрес электронной почты'],
    'website':        ['Web-site', 'website', 'Сайт'],
    'license_status': ['Status of license', 'license_status', 'Статус лицензии'],
    'license_date':   ['Date of issue of license', 'license_date', 'Дата выдачи лицензии'],
    'license_expiry': ['Expiry date of license', 'license_expiry', 'Срок окончания действия лицензии'],
    'ownership':      ['Type of ownership', 'ownership', 'Признак типа учреждения'],
}

# Фолбэк — позиции колонок в исходном файле (A=0 … AA=26)
POSITION_FALLBACK = {
    'name': 0, 'admin_area': 2, 'phone': 9, 'email': 10, 'website': 11,
    'license_status': 16, 'license_date': 20, 'license_expiry': 21, 'ownership': 23,
}


def load_table(path, sheet=None, header_row=2):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Файл не найден: {p}")
    suffix = p.suffix.lower()
    try:
        if suffix in ('.xlsx', '.xls', '.xlsm'):
            df = pd.read_excel(p, sheet_name=sheet if sheet is not None else 0,
                               header=header_row)
        elif suffix == '.csv':
            df = pd.read_csv(p, header=header_row,
                             encoding='utf-8-sig', sep=None, engine='python')
        else:
            sys.exit(f"Неподдерживаемый формат: {suffix}")
    except Exception as e:
        sys.exit(f"Ошибка чтения файла: {e}")
    df = df.dropna(how='all').reset_index(drop=True)
    return df


def resolve_columns(df):
    mapping = {}
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for logical, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if a.strip().lower() in lower_map:
                mapping[logical] = lower_map[a.strip().lower()]
                break
    for logical, idx in POSITION_FALLBACK.items():
        if logical not in mapping and idx < df.shape[1]:
            mapping[logical] = df.columns[idx]
    missing = [k for k in FIELD_ALIASES if k not in mapping]
    if missing:
        print(f"⚠ Не найдены колонки: {missing}. Будут пропущены соответствующие признаки.")
    return mapping


# ============================ 2. Парсинг полей ============================

_KV_RE = re.compile(r'([A-Za-z][A-Za-z0-9_]*)=([^,}\]]*)')


def count_key(s, key):
    if pd.isna(s):
        return 0
    return str(s).count(f"{key}=")


def nonempty_values(s, key):
    if pd.isna(s):
        return []
    out = []
    for m in _KV_RE.finditer(str(s)):
        if m.group(1) == key:
            v = m.group(2).strip()
            if v and v.lower() not in ('none', 'nan'):
                out.append(v)
    return out


def has_nonempty(s, key):
    return bool(nonempty_values(s, key))


def parse_date_to_year(d):
    if pd.isna(d):
        return np.nan
    s = str(d).strip()
    if not s:
        return np.nan
    dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
    return dt.year if pd.notna(dt) else np.nan


# ============================ 3. Инженерия признаков ============================

def build_features(df):
    cm = resolve_columns(df)
    f = pd.DataFrame(index=df.index)

    def col(name):
        return df[cm[name]] if name in cm else pd.Series([np.nan]*len(df), index=df.index)

    f['admin_area']     = col('admin_area').astype(str).str.strip().replace({'nan': np.nan, '': np.nan})
    f['ownership']      = col('ownership').astype(str).str.strip().str.lower().replace({'nan': np.nan, '': np.nan})
    f['license_status'] = col('license_status').astype(str).str.strip().replace({'nan': np.nan, '': np.nan})

    f['n_phones'] = col('phone').apply(lambda x: count_key(x, 'PublicPhone'))
    f['n_emails'] = col('email').apply(lambda x: count_key(x, 'Email'))

    f['has_website'] = col('website').apply(lambda x: has_nonempty(x, 'WebSite')).astype(int)
    f['has_email']   = (f['n_emails'] > 0).astype(int)
    f['has_phone']   = (f['n_phones'] > 0).astype(int)

    f['license_year'] = col('license_date').apply(parse_date_to_year)

    expiry = col('license_expiry').astype(str).str.lower()
    f['is_perpetual_license'] = expiry.apply(
        lambda s: 1 if 'бессроч' in s else (0 if s.strip() not in ('', 'nan') else np.nan)
    )

    f['name_len'] = col('name').astype(str).str.len()
    return f


# ============================ 4. Статистика ============================

def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.values.sum()
    r, k = confusion_matrix.shape
    denom = n * (min(r, k) - 1)
    return np.nan if denom <= 0 else np.sqrt(chi2 / denom)


def report_distribution(series, name):
    print(f"\n--- Распределение: {name} ---")
    vc = series.dropna().value_counts()
    total = vc.sum()
    if total == 0:
        print("  Нет данных.")
        return
    for k, v in vc.items():
        print(f"  {str(k)[:70]:<70} {v:>5}  ({100*v/total:5.1f}%)")
    print(f"  Всего (без NA): {total}")


def report_cat_cat(feats, a, b):
    print(f"\n--- {a} × {b} ---")
    sub = feats[[a, b]].dropna()
    if sub.empty:
        print("  Нет данных."); return
    ct = pd.crosstab(sub[a], sub[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print("  Недостаточно категорий для χ²."); return
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    v = cramers_v(ct)
    print(f"  n = {ct.values.sum()}, χ² = {chi2:.3f}, df = {dof}, "
          f"p = {p:.4g}, V Крамера = {v:.3f}")
    print("  Таблица сопряжённости:")
    print(ct.to_string())


def report_num_num(feats, a, b):
    print(f"\n--- {a} × {b} ---")
    sub = feats[[a, b]].dropna()
    if len(sub) < 3:
        print("  Недостаточно данных."); return
    x, y = sub[a].astype(float), sub[b].astype(float)
    if x.nunique() < 2 or y.nunique() < 2:
        print("  Один из признаков константен — корреляция не определена."); return
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    print(f"  n = {len(sub)}")
    print(f"  Пирсон:   r = {pr:+.3f}, p = {pp:.4g}")
    print(f"  Спирмен:  ρ = {sr:+.3f}, p = {sp:.4g}")


def report_cat_num(feats, cat, num):
    print(f"\n--- {cat} × {num} ---")
    sub = feats[[cat, num]].dropna()
    if sub.empty:
        print("  Нет данных."); return
    grp_stats = sub.groupby(cat)[num].agg(['count', 'mean', 'median', 'std'])
    print(f"  n = {len(sub)}")
    print(grp_stats.round(2).to_string())
    groups = [g[num].astype(float).values for _, g in sub.groupby(cat)]
    if len(groups) >= 2:
        try:
            h, p = stats.kruskal(*groups)
            print(f"  Kruskal–Wallis: H = {h:.3f}, p = {p:.4g}")
        except ValueError as e:
            print(f"  Kruskal–Wallis не выполнен: {e}")
    if len(groups) == 2:
        codes = sub[cat].astype('category').cat.codes.astype(float)
        r, p = stats.pearsonr(codes, sub[num].astype(float))
        print(f"  Point-biserial: r = {r:+.3f}, p = {p:.4g}")


# ============================ 5. Основной сценарий ============================

def main():
    ap = argparse.ArgumentParser(description="Корреляционный анализ вузов Москвы (data.mos.ru, 3326).")
    ap.add_argument('path', help='Путь к .xlsx или .csv файлу')
    ap.add_argument('--header-row', type=int, default=2,
                    help='0-based индекс строки с техническими именами полей (по умолчанию 2 = 3-я строка). '
                         'Для файла с одной строкой-заголовком укажите 0.')
    ap.add_argument('--sheet', default=None, help='Имя листа (xlsx), по умолчанию первый')
    args = ap.parse_args()

    print("=" * 72)
    print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ДАННЫХ ВУЗОВ МОСКВЫ (data.mos.ru, 3326)")
    print("=" * 72)

    df = load_table(args.path, sheet=args.sheet, header_row=args.header_row)
    print(f"Загружено строк: {len(df)}, колонок: {df.shape[1]}")
    print("Колонки:", list(df.columns[:12]), "…" if df.shape[1] > 12 else "")

    feats = build_features(df)

    print("\n--- Покрытие признаков ---")
    for c in feats.columns:
        n = feats[c].notna().sum()
        print(f"  {c:<22} {n:>4} / {len(feats)}  ({100*n/len(feats):5.1f}%)")

    # 3.1. Распределения
    report_distribution(feats['admin_area'], 'Административный округ')
    report_distribution(feats['ownership'], 'Тип собственности')
    report_distribution(feats['license_status'], 'Статус лицензии')
    report_distribution(feats['is_perpetual_license'].map({1.0: 'бессрочно', 0.0: 'ограниченный срок'}),
                        'Срок действия лицензии')
    report_distribution(feats['has_website'].map({1: 'есть', 0: 'нет'}), 'Наличие сайта')
    report_distribution(feats['has_email'].map({1: 'есть', 0: 'нет'}), 'Наличие e-mail')

    # 3.2. Категориальные × категориальные
    for a, b in [
        ('admin_area', 'ownership'),
        ('admin_area', 'license_status'),
        ('ownership', 'license_status'),
        ('ownership', 'has_website'),
    ]:
        report_cat_cat(feats, a, b)

    # 3.3. Числовые × числовые
    for a, b in [
        ('n_phones', 'n_emails'),
        ('n_phones', 'name_len'),
        ('n_emails', 'name_len'),
        ('license_year', 'name_len'),
    ]:
        report_num_num(feats, a, b)

    # 3.4. Категориальные × числовые
    for cat, num in [
        ('ownership', 'license_year'),
        ('ownership', 'n_phones'),
        ('ownership', 'n_emails'),
        ('ownership', 'name_len'),
        ('admin_area', 'n_phones'),
        ('admin_area', 'n_emails'),
        ('admin_area', 'license_year'),
    ]:
        report_cat_num(feats, cat, num)

    # 3.5. Сводная матрица Спирмена
    print("\n--- Матрица корреляций (Spearman) ---")
    num_cols = ['n_phones', 'n_emails', 'license_year', 'name_len', 'has_website', 'has_email']
    corr = feats[num_cols].astype(float).corr(method='spearman')
    print(corr.round(3).to_string())

    print("\nГотово.")


if __name__ == '__main__':
    main()