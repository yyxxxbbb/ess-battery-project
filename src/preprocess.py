"""
데이터 로딩 및 정리
- 원본 .mat(MATLAB v7.3 = HDF5)을 셀 단위로 읽어 필요한 값만 추출
- Batch 1·2 이어 붙이기, EOL 미도달 셀 제거
- 결과를 배치별 pkl로 저장

사용법:
    python src/preprocess.py --data_dir data/raw --out_dir data/processed
"""
import argparse
import gc
import glob
import os
import pickle
import sys

import h5py
import numpy as np

SUMMARY_KEYS = ['IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmin', 'Tmax', 'chargetime', 'cycle']
BATCH_FILES = {'b1': '2017-05-12', 'b2': '2018-02-20', 'b3': '2018-04-12'}


def to_float(x):
    return float(np.ravel(x)[0])


def load_batch(mat_path, prefix, max_cycle=101):
    """셀 단위로 .mat을 읽어 summary 전체 + 초기 max_cycle개 사이클의 Qdlin/Tdlin만 저장"""
    data = {}
    with h5py.File(mat_path, 'r') as f:
        batch = f['batch']
        n_cells = batch['summary'].shape[0]
        for i in range(n_cells):
            cycle_life = f[batch['cycle_life'][i, 0]][()]
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
            s = f[batch['summary'][i, 0]]
            summary = {k: np.hstack(s[k][0, :].tolist()) for k in SUMMARY_KEYS}
            cycles = f[batch['cycles'][i, 0]]
            n_cyc = min(cycles['I'].shape[0], max_cycle)
            cyc = {}
            for j in range(n_cyc):
                cyc[str(j)] = {
                    'Qdlin': np.hstack(f[cycles['Qdlin'][j, 0]][()]),
                    'Tdlin': np.hstack(f[cycles['Tdlin'][j, 0]][()]),
                }
            data[f'{prefix}c{i}'] = {'cycle_life': cycle_life, 'charge_policy': policy,
                                     'summary': summary, 'cycles': cyc}
    gc.collect()
    return data


def merge_batch1_batch2(b1, b2):
    """Batch 1의 일부 셀은 Batch 2 실험에서 이어서 측정됨 → 이어 붙이고 Batch 2에서 제거.
    EOL(80% 용량)에 도달하지 못한 Batch 1 셀도 제거. (원저자 공개 코드 기준 인덱스)"""
    b1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    b2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    add_len = [662, 981, 1060, 208, 482]
    for i, bk in enumerate(b1_keys):
        b1[bk]['cycle_life'] = b1[bk]['cycle_life'] + add_len[i]
        for k in b1[bk]['summary']:
            tail = b2[b2_keys[i]]['summary'][k]
            if k == 'cycle':
                tail = tail + len(b1[bk]['summary'][k])
            b1[bk]['summary'][k] = np.hstack((b1[bk]['summary'][k], tail))
    for k in b2_keys:
        del b2[k]
    for k in ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']:
        b1.pop(k, None)
    return b1, b2


def drop_missing_cycle_life(data):
    """cycle_life가 NaN인(정의 불가) 셀 제거"""
    drop = [k for k, v in data.items() if not np.isfinite(to_float(v['cycle_life']))]
    for k in drop:
        del data[k]
    return data, drop


def find_mat(data_dir, date_prefix):
    hits = glob.glob(os.path.join(data_dir, '**', f'{date_prefix}_batchdata*.mat'), recursive=True)
    if not hits:
        raise FileNotFoundError(f'{date_prefix} .mat 파일을 {data_dir}에서 찾을 수 없음')
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/raw')
    ap.add_argument('--out_dir', default='data/processed')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    batches = {b: load_batch(find_mat(args.data_dir, d), b) for b, d in BATCH_FILES.items()}
    batches['b1'], batches['b2'] = merge_batch1_batch2(batches['b1'], batches['b2'])
    for b, d in batches.items():
        d, dropped = drop_missing_cycle_life(d)
        with open(os.path.join(args.out_dir, f'batch_{b}.pkl'), 'wb') as fp:
            pickle.dump(d, fp)
        print(f'{b}: {len(d)} cells (NaN 제거: {dropped})')


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    main()
