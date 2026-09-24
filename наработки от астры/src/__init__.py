# -*- coding: utf-8 -*-
"""Гибридный сервис контроля качества DXA (ДРА).

Пакет объединяет лучшее из двух решений и базового кода команды:

  * :mod:`src.config`     — единая конфигурация (таксономия, v3-источники, флаги);
  * :mod:`src.preprocess` — пиксельный конвейер (bone window + denoise + CLAHE);
  * :mod:`src.dicom_io`   — чтение/дедуп/де-ид/область/масштаб пикселя;
  * :mod:`src.features`   — геометрические признаки (вкл. spine_midline_residual);
  * :mod:`src.models`     — мультизадачная сеть, бэкбоны, лоссы, TTA;
  * :mod:`src.data`       — манифест, dataset/сэмплер, синтетические артефакты;
  * :mod:`src.metrics`    — пороги (f1/prior/blend), ДИ, метка организатора;
  * :mod:`src.stack`      — гибридный стекер CNN + геометрия (конфигурация v3);
  * :mod:`src.train`      — честная CV (StratifiedGroupKFold по study_uid);
  * :mod:`src.calibrate`  — подбор порогов по OOF;
  * :mod:`src.evaluate`   — BA / Macro-F1 / ROC-AUC с 95% ДИ;
  * :mod:`src.inference`  — пакетный инференс DICOM -> отчёт по ТЗ;
  * :mod:`src.predict`    — CLI пакетной обработки.
"""

__version__ = "1.0.0-hybrid"