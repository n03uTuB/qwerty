# -*- coding: utf-8 -*-
"""Общие пути для экспериментов (папка dxa_qc_work — отдельно от репозитория mogaem).

Репозиторий и данные остаются в mogaem и НЕ изменяются: отсюда мы только читаем
артефакты и код. Все свои скрипты и выводы держим в dxa_qc_work.

Переопределить репозиторий можно переменной окружения DXA_REPO.
"""
from __future__ import annotations

import os
import sys

REPO = os.environ.get("DXA_REPO", r"C:\Users\куку\Desktop\mogaem")
DXA_QC = os.path.join(REPO, "dxa_qc")
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "out")

# чтобы импортировался пакет src из репозитория
if DXA_QC not in sys.path:
    sys.path.insert(0, DXA_QC)
os.makedirs(OUT, exist_ok=True)
