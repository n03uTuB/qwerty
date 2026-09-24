# -*- coding: utf-8 -*-
"""Разбор реальных нарушений «предметы/артефакты/наложения» по исходной разметке."""
import os

import pandas as pd

F = os.path.join(os.environ.get(
    "DXA_DATASET", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "_dsroot")), "разметка.xlsx")
df = pd.read_excel(F, sheet_name="Калибровка", header=0)
sub = df.iloc[1:].copy()
sub.columns = ["No", "study", "spine_pos", "spine_axis", "artifacts",
               "femR_pos", "femR_roi", "femL_pos", "femL_roi",
               "itog_sp", "itog_R", "itog_L", "comment",
               "u13", "obsh_sp", "obsh_R", "obsh_L", "u17", "u18"]

art = sub[sub["artifacts"] == 1]
print("строк с artifacts=1:", len(art))
for _, r in art.iterrows():
    print("  study=%-32s comment=%s" % (str(r["study"])[:32], r["comment"]))
print()
print("--- комментарии среди artifacts=1 ---")
print(art["comment"].value_counts(dropna=False).to_string())
print()
print("--- все непустые комментарии (любые метки) ---")
c = sub["comment"].dropna()
print(c.value_counts().to_string())
