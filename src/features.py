"""
셀 단위 피처 생성 (EDA와 모델링 공통 사용)
- ΔQ(V) = Q100(V) - Q10(V) 통계량
- 초기 100 사이클 방전 용량 / 저항 / 온도 / 충전 시간 요약값
- 충전 프로토콜 첫 단계 C-rate

사용법:
    python src/features.py --in_dir data/processed --out data/processed/features.csv
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

BATCHES = ['b1', 'b2', 'b3']
V = np.linspace(3.6, 2.0, 1000)  # Qdlin 전압 축


def to_float(x):
    return float(np.ravel(x)[0])


def cyc_key(cell, n):
    """사이클 번호 n -> cycles 딕셔너리 key (summary cycle 시작값 기준 보정)"""
    offset = int(cell['summary']['cycle'][0])
    return str(n - offset)


def series(cell, key, lo=None, hi=None):
    """summary 값을 사이클 번호 인덱스 Series로 변환, 범위 밖 노이즈 제거"""
    sm = cell['summary']
    s = pd.Series(sm[key], index=sm['cycle'].astype(int))
    if lo is not None:
        s = s[(s > lo) & (s < hi)]
    return s


def delta_q(cell, hi=100, lo=10):
    return cell['cycles'][cyc_key(cell, hi)]['Qdlin'] - cell['cycles'][cyc_key(cell, lo)]['Qdlin']


def first_crate(policy):
    m = re.findall(r'([\d.]+)C', policy)
    return float(m[0]) if m else np.nan


def cell_features(batch, cid, cell):
    cl = to_float(cell['cycle_life'])
    qd = series(cell, 'QDischarge', 0.8, 1.2)
    ir = series(cell, 'IR', 0, 1)
    ct = series(cell, 'chargetime', 0, 60)
    tmax = series(cell, 'Tmax', 0, 80)
    tavg = series(cell, 'Tavg', 0, 80)
    q_early = qd.loc[2:100]
    d = delta_q(cell)
    return {
        'batch': batch, 'cell': cid, 'policy': cell['charge_policy'],
        'C1': first_crate(cell['charge_policy']),
        'cycle_life': cl, 'log_cl': np.log10(cl),
        # ΔQ(V) 피처
        'dq_logvar': np.log10(np.var(d)),
        'dq_logmin': np.log10(np.abs(d.min()) + 1e-12),
        'dq_mean': d.mean(), 'dq_skew': skew(d), 'dq_kurt': kurtosis(d),
        # 방전 용량 피처
        'qd_2': qd.loc[1:3].median(),
        'qd_max_minus_2': q_early.max() - qd.loc[1:3].median(),
        'qd_100_minus_2': qd.loc[98:100].median() - qd.loc[1:3].median(),
        'qd_slope_2_100': np.polyfit(q_early.index, q_early.values, 1)[0] * 1000,  # mAh/cycle
        # 저항, 온도, 충전 시간
        'ir_2': ir.loc[1:3].median(), 'ir_min_2_100': ir.loc[2:100].min(),
        'ir_100_minus_2': ir.loc[98:100].median() - ir.loc[1:3].median(),
        'tmax_mean_2_100': tmax.loc[2:100].mean(), 'tavg_mean_2_100': tavg.loc[2:100].mean(),
        'ct_mean_2_6': ct.loc[2:6].mean(),
    }


def build_features(data):
    """data: {'b1': {cell_id: cell_dict}, 'b2': ..., 'b3': ...}"""
    rows = [cell_features(b, cid, cell) for b in BATCHES for cid, cell in data[b].items()]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='data/processed')
    ap.add_argument('--out', default='data/processed/features.csv')
    args = ap.parse_args()
    data = {}
    for b in BATCHES:
        with open(os.path.join(args.in_dir, f'batch_{b}.pkl'), 'rb') as fp:
            data[b] = pickle.load(fp)
    feat = build_features(data)
    feat.to_csv(args.out, index=False)
    print(feat.groupby('batch').size())


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    main()
