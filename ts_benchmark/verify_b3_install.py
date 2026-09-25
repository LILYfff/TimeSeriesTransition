import inspect
import os
from ts_benchmark.baselines.dag.hct_retrieval import HCTRetriever

print('Imported HCTRetriever from:')
print(os.path.abspath(inspect.getfile(HCTRetriever)))
print('\nConstructor signature:')
print(inspect.signature(HCTRetriever.__init__))

sig = inspect.signature(HCTRetriever.__init__).parameters
required = ['bins', 'preselect', 'future_lambda']
missing = [x for x in required if x not in sig]
if missing:
    raise RuntimeError(f'B3 install mismatch: missing constructor args: {missing}')

r = HCTRetriever(
    seq_len=96,
    pred_len=96,
    series_dim=1,
    topk=5,
    stride=12,
    mode=3,
    bins=24,
    preselect=20,
    future_lambda=0.0,
)
print('\nB3 constructor smoke test: PASS')
print(f'bins={r.bins}, preselect={r.preselect}, future_lambda={r.future_lambda}')
