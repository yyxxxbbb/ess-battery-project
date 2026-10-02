"""
모델 학습 및 평가 (DAY 1 모델 설계 전략 구현)

[분할]
- Batch 1을 충전 프로토콜 단위로 나눔 (GroupShuffleSplit, 20%) → Train / Valid(Hold-out)
- Train 안에서 프로토콜 그룹 기준 GroupKFold(5)로 하이퍼파라미터 탐색 → Train(CV) 성능
- 선택된 하이퍼파라미터로 Batch 1 전체 재학습 → Test(Batch 2), 추가 Test(Batch 3)

[Target / Metric]
- Target: log10(cycle_life), MAPE는 원 단위(cycle)로 역변환해서 계산
- Gap은 "후단계 오차 - 전단계 오차" (MAPE는 낮을수록 좋으므로 (+) = 성능 저하)

사용법:
    python src/train.py --features data/processed/features.csv --out_dir results
노트북:
    이 파일 내용을 셀에 붙여넣고  out = run_all(feat, out_dir='/kaggle/working/results')
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
PAPER_MAPE = 9.1                 # 원논문 Regression 테스트 오차 (%)
MAIN_MODEL = 'ElasticNet'
FEATURES_VARIANCE = ['dq_logvar']                                   # 베이스라인 (원논문 variance model 형태)
FEATURES_MAIN = ['dq_logvar', 'dq_logmin', 'dq_mean', 'dq_kurt',    # DAY 1 Feature Engineering 결과
                 'qd_2', 'qd_max_minus_2']


# ------------------------------------------------------------------ metric
def mape_log(y_true_log, y_pred_log):
    """log10 스케일 값을 받아 원 단위(cycle)로 되돌린 뒤 MAPE(%) 계산"""
    t = 10 ** np.asarray(y_true_log, dtype=float)
    p = 10 ** np.asarray(y_pred_log, dtype=float)
    return float(np.mean(np.abs(p - t) / t) * 100)


MAPE_SCORER = make_scorer(mape_log, greater_is_better=False)
_LGBM_WARNED = False


# ------------------------------------------------------------------ models
def get_models():
    """{이름: (estimator, param_grid, 사용 피처)}"""
    alphas = np.logspace(-4, 0, 9)
    models = {
        'Linear (dq_logvar)': (make_pipeline(StandardScaler(), LinearRegression()),
                               {}, FEATURES_VARIANCE),
        'Ridge': (make_pipeline(StandardScaler(), Ridge()),
                  {'ridge__alpha': np.logspace(-3, 2, 11)}, FEATURES_MAIN),
        'Lasso': (make_pipeline(StandardScaler(), Lasso(max_iter=50000)),
                  {'lasso__alpha': alphas}, FEATURES_MAIN),
        'ElasticNet': (make_pipeline(StandardScaler(), ElasticNet(max_iter=50000)),
                       {'elasticnet__alpha': alphas,
                        'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}, FEATURES_MAIN),
        'RandomForest': (RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED),
                         {'max_depth': [2, 3, None]}, FEATURES_MAIN),
    }
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = (LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7,
                                            min_child_samples=3, random_state=SEED, verbose=-1),
                              {'max_depth': [2, 3]}, FEATURES_MAIN)
    except ImportError:
        global _LGBM_WARNED
        if not _LGBM_WARNED:
            print('[info] lightgbm 미설치 → LightGBM 비교 생략')
            _LGBM_WARNED = True
    return models


# ------------------------------------------------------------------ split
def split_batch1(feat, test_size=0.2, seed=SEED):
    """Batch 1을 충전 프로토콜 단위로 Train / Hold-out 분리 (같은 프로토콜 셀이 양쪽에 섞이지 않음)"""
    b1 = feat[feat.batch == 'b1'].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, ho_idx = next(gss.split(b1, groups=b1['policy']))
    return b1, b1.iloc[tr_idx].reset_index(drop=True), b1.iloc[ho_idx].reset_index(drop=True)


def _clean_params(params):
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()}


# ------------------------------------------------------------------ evaluate
def evaluate_model(name, estimator, grid, cols, b1, train, holdout, tests, n_splits=5):
    gs = GridSearchCV(estimator, grid, scoring=MAPE_SCORER, cv=GroupKFold(n_splits=n_splits), refit=True)
    gs.fit(train[cols], train['log_cl'], groups=train['policy'])

    res = {'model': name, 'features': ', '.join(cols),
           'best_params': json.dumps(_clean_params(gs.best_params_)),
           'Train (Batch 1 CV)': -gs.best_score_,
           'Train fit (참고)': mape_log(train['log_cl'], gs.predict(train[cols])),
           'Valid (Batch 1 Hold-out)': mape_log(holdout['log_cl'], gs.predict(holdout[cols]))}

    final = clone(gs.best_estimator_).fit(b1[cols], b1['log_cl'])   # Batch 1 전체로 재학습
    preds = {}
    for tname, tdf in tests.items():
        p = final.predict(tdf[cols])
        res[f'Test ({tname})'] = mape_log(tdf['log_cl'], p)
        preds[tname] = p
    return res, final, preds


def add_gaps(df):
    df['Gap (Train-Valid)'] = df['Valid (Batch 1 Hold-out)'] - df['Train (Batch 1 CV)']
    df['Gap (Valid-Test)'] = df['Test (Batch 2)'] - df['Valid (Batch 1 Hold-out)']
    df['Gap (Target-Test)'] = df['Test (Batch 2)'] - PAPER_MAPE
    df['Gap (Batch2-Batch3)'] = df['Test (Batch 3)'] - df['Test (Batch 2)']
    df['Gap (Target-Test, Batch 3)'] = df['Test (Batch 3)'] - PAPER_MAPE
    return df


def performance_table(row):
    """과제 Reporting format (Batch 3 추가 포함) 형태의 표"""
    r = lambda k: round(float(row[k]), 2)
    return pd.DataFrame([
        ['Train (Batch 1 CV)', r('Train (Batch 1 CV)'), '프로토콜 그룹 GroupKFold(5) 평균'],
        ['Valid (Batch 1 Hold-out)', r('Valid (Batch 1 Hold-out)'), '프로토콜 단위 20% 분리'],
        ['Test (Batch 2)', r('Test (Batch 2)'), 'Batch 1 전체 재학습 후 평가'],
        ['Gap (Train-Valid)', r('Gap (Train-Valid)'), '(+) : 과적합 의심'],
        ['Gap (Valid-Test)', r('Gap (Valid-Test)'), '(+) : 배치간 일반화 저하 의심'],
        ['Gap (Target-Test)', r('Gap (Target-Test)'), f'Target : 원논문 {PAPER_MAPE}% / (+) : 논문 대비 미달'],
        ['Test (Batch 3)', r('Test (Batch 3)'), ''],
        ['Gap (Batch2-Batch3)', r('Gap (Batch2-Batch3)'), 'Test 성능 간 비교'],
        ['Gap (Target-Test, Batch 3)', r('Gap (Target-Test, Batch 3)'), 'Batch 3 기준, 원논문 성능 비교'],
    ], columns=['구분', 'MAPE (%)', '비고'])


# ------------------------------------------------------------------ error analysis
def build_predictions(final, cols, b1, tests, preds):
    frames = [b1.assign(pred_log=final.predict(b1[cols]), split='Batch 1 (fit)')]
    for tname, tdf in tests.items():
        frames.append(tdf.assign(pred_log=preds[tname], split=tname))
    df = pd.concat(frames, ignore_index=True)
    df['pred_cycle_life'] = 10 ** df['pred_log']
    df['error(%)'] = (df['pred_cycle_life'] - df['cycle_life']) / df['cycle_life'] * 100  # (+) 과대 예측
    df['APE(%)'] = df['error(%)'].abs()
    lo, hi = b1['cycle_life'].min(), b1['cycle_life'].max()
    flo, fhi = b1['dq_logvar'].min(), b1['dq_logvar'].max()
    df['target_out_of_b1_range'] = (df['cycle_life'] < lo) | (df['cycle_life'] > hi)
    df['dq_logvar_out_of_b1_range'] = (df['dq_logvar'] < flo) | (df['dq_logvar'] > fhi)
    return df


def error_analysis(pred_df, top_n=5):
    test = pred_df[pred_df.split != 'Batch 1 (fit)']
    cols = ['split', 'cell', 'policy', 'cycle_life', 'pred_cycle_life', 'error(%)', 'dq_logvar',
            'target_out_of_b1_range', 'dq_logvar_out_of_b1_range']
    top = test.sort_values('APE(%)', ascending=False).groupby('split').head(top_n)[cols]
    by_range = test.groupby(['split', 'target_out_of_b1_range'])['APE(%)'].agg(['count', 'mean']).round(2)
    bias = test.groupby('split')['error(%)'].agg(['mean', 'median']).round(2)   # (+) 과대 / (-) 과소
    by_policy = (test.groupby(['split', 'policy'])['APE(%)'].agg(['count', 'mean'])
                 .sort_values('mean', ascending=False).round(2))
    return {'top_errors': top, 'by_range': by_range, 'bias': bias, 'by_policy': by_policy}


# ------------------------------------------------------------------ plots
def plot_pred_vs_actual(pred_df, path, title):
    colors = {'Batch 1 (fit)': 'tab:blue', 'Batch 2': 'tab:orange', 'Batch 3': 'tab:green'}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for s, g in pred_df.groupby('split'):
        ax[0].scatter(g.cycle_life, g.pred_cycle_life, s=22, label=s, color=colors.get(s))
        ax[1].scatter(g.cycle_life, g['error(%)'], s=22, label=s, color=colors.get(s))
    lim = [300, 2500]
    ax[0].plot(lim, lim, 'k--', lw=1)
    ax[0].fill_between(lim, [v * 0.85 for v in lim], [v * 1.15 for v in lim], color='gray', alpha=.15,
                       label='±15%')
    ax[0].set(xscale='log', yscale='log', xlim=lim, ylim=lim, xlabel='actual cycle life',
              ylabel='predicted cycle life', title=f'{title}: predicted vs actual')
    ax[1].axhline(0, c='k', lw=1)
    ax[1].set(xscale='log', xlabel='actual cycle life', ylabel='error (%) [+ over / - under]',
              title=f'{title}: error vs actual life')
    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_model_comparison(comp, path):
    cols = ['Train (Batch 1 CV)', 'Valid (Batch 1 Hold-out)', 'Test (Batch 2)', 'Test (Batch 3)']
    ax = comp.set_index('model')[cols].plot.bar(figsize=(12, 5), rot=0)
    ax.axhline(PAPER_MAPE, ls='--', c='r', lw=1, label=f'paper {PAPER_MAPE}%')
    ax.set(ylabel='MAPE (%)', title='Model comparison (lower is better)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------ extra checks
def coefficient_table(final, cols):
    """표준화 계수: 어떤 피처가 얼마나, 어느 방향으로 쓰였는지 (0 = L1으로 제외됨)"""
    est = final[-1]
    if not hasattr(est, 'coef_'):
        return None
    return (pd.DataFrame({'feature': cols, 'std_coef': est.coef_})
            .assign(abs_coef=lambda d: d.std_coef.abs()).sort_values('abs_coef', ascending=False)
            .drop(columns='abs_coef').round(4))


def holdout_stability(feat, model_name=MAIN_MODEL, seeds=range(20)):
    """Hold-out이 9개 안팎의 셀이라 분할에 따라 성능이 흔들림 → seed를 바꿔 반복해 범위 확인"""
    est, grid, cols = get_models()[model_name]
    tests = {'Batch 2': feat[feat.batch == 'b2'].reset_index(drop=True),
             'Batch 3': feat[feat.batch == 'b3'].reset_index(drop=True)}
    rows = []
    for s in seeds:
        b1, tr, ho = split_batch1(feat, seed=s)
        res, _, _ = evaluate_model(model_name, est, grid, cols, b1, tr, ho, tests)
        rows.append({'seed': s, 'n_holdout': len(ho), **{k: res[k] for k in
                     ['Train (Batch 1 CV)', 'Valid (Batch 1 Hold-out)', 'Test (Batch 2)', 'Test (Batch 3)']}})
    df = pd.DataFrame(rows)
    return df, df.drop(columns=['seed']).describe().loc[['mean', 'std', 'min', 'max']].round(2)


# ------------------------------------------------------------------ main pipeline
def run_all(feat, out_dir='results', seed=SEED, stability=True):
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    feat = feat.dropna(subset=FEATURES_MAIN + ['log_cl']).copy()

    b1, train, holdout = split_batch1(feat, seed=seed)
    tests = {'Batch 2': feat[feat.batch == 'b2'].reset_index(drop=True),
             'Batch 3': feat[feat.batch == 'b3'].reset_index(drop=True)}
    print(f'Train {len(train)} cells ({train.policy.nunique()} policies) | '
          f'Hold-out {len(holdout)} cells ({holdout.policy.nunique()} policies) | '
          f'Batch 2 {len(tests["Batch 2"])} | Batch 3 {len(tests["Batch 3"])}')
    overlap = set(train.policy) & set(holdout.policy)
    print('Train/Hold-out 프로토콜 겹침:', overlap if overlap else '없음 (누수 없음)')

    rows, finals, all_preds = [], {}, {}
    for name, (est, grid, cols) in get_models().items():
        res, final, preds = evaluate_model(name, est, grid, cols, b1, train, holdout, tests)
        rows.append(res)
        finals[name], all_preds[name] = final, preds
    comp = add_gaps(pd.DataFrame(rows))
    comp.round(2).to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False, encoding='utf-8-sig')

    main_row = comp[comp.model == MAIN_MODEL].iloc[0]
    perf = performance_table(main_row)
    perf.to_csv(os.path.join(out_dir, 'model_performance.csv'), index=False, encoding='utf-8-sig')

    cols = get_models()[MAIN_MODEL][2]
    pred_df = build_predictions(finals[MAIN_MODEL], cols, b1, tests, all_preds[MAIN_MODEL])
    pred_df.round(4).to_csv(os.path.join(out_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')
    errors = error_analysis(pred_df)
    coefs = coefficient_table(finals[MAIN_MODEL], cols)

    plot_model_comparison(comp, os.path.join(fig_dir, 'model_comparison.png'))
    plot_pred_vs_actual(pred_df, os.path.join(fig_dir, 'pred_vs_actual_elasticnet.png'), MAIN_MODEL)
    # 트리 모델의 외삽 한계 확인용 비교 그림
    tree = 'LightGBM' if 'LightGBM' in finals else 'RandomForest'
    tree_df = build_predictions(finals[tree], get_models()[tree][2], b1, tests, all_preds[tree])
    plot_pred_vs_actual(tree_df, os.path.join(fig_dir, f'pred_vs_actual_{tree.lower()}.png'), tree)

    out = {'comparison': comp, 'performance': perf, 'predictions': pred_df,
           'errors': errors, 'coefficients': coefs, 'tree_predictions': tree_df}
    if stability:
        stab_raw, stab = holdout_stability(feat)
        stab_raw.to_csv(os.path.join(out_dir, 'holdout_stability.csv'), index=False)
        out['stability'] = stab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='data/processed/features.csv')
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()
    out = run_all(pd.read_csv(args.features), out_dir=args.out_dir)
    print(out['performance'].to_string(index=False))


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    main()
