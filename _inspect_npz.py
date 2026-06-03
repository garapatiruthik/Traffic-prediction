import numpy as np
sp = np.load('data/processed/scaler_params.npz', allow_pickle=True)
print('Keys:', list(sp.keys()))
for k in sp.keys():
    v = sp[k]
    print(f'  {k}: type={type(v).__name__}  shape={getattr(v,"shape",None)}')
    try:
        print(f'    value={v}')
    except Exception as e:
        print(f'    (repr) {v!r}')
