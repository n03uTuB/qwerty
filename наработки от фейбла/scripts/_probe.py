import importlib
for m in ['torchvision', 'timm', 'albumentations', 'PIL', 'cv2', 'skimage',
          'xgboost', 'lightgbm', 'catboost', 'torch']:
    try:
        mod = importlib.import_module(m)
        print('%-14s OK  %s' % (m, getattr(mod, '__version__', '?')))
    except Exception as e:
        print('%-14s MISSING (%s)' % (m, type(e).__name__))
