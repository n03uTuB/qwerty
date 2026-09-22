"""Параметры ИИ-сервиса для оформления результата по «Базовым функциональным требованиям»
ЦДТ (версия от 15.09.2026): https://mosmed.ai/ai/docs/
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_UID_LENGTH = 64


@dataclass(frozen=True)
class ServiceConfig:
    name: str = "DXA-QC"  # идентификатор ИИ-сервиса в ЕРИС ЕМИАС (выдаётся при подключении)
    version: str = "0.2.0"
    model_id: int = 1000  # номер модели сервиса (modelId), выдаётся при подключении
    # Направление в названии дополнительной серии (п. 3.4). Для DXA в прил. 13 аббревиатуры нет — согласовать.
    direction: str = "DXA-QC"
    purpose: str = (
        "Оценка качества выполнения рентгеновской денситометрии "
        "и корректности разметки анатомических структур"
    )
    registered_medical_device: bool = False
    integration: str = "mosmedii"  # "pum" — ЕМИАС.ЕРИС.ПУМ, "mosmedii" — ЕМИАС.ЕРИС.ПУМ.МОСМЕДИИ (прил. 7)

    @property
    def series_description(self) -> str:
        """Рекомендуемый формат п. 3.4: «название ИИ-сервиса_направление»."""
        return f"{self.name}_{self.direction}"

    @property
    def conclusion_warning(self) -> str:  # п. 3.3
        if self.registered_medical_device:
            return "Заключение подготовлено медицинским изделием с применением технологий искусственного интеллекта"
        return "Заключение подготовлено программным обеспечением с применением технологий искусственного интеллекта"

    @property
    def usage_warning(self) -> str:  # п. 3.3
        return "Для поддержки принятия врачебных решений" if self.registered_medical_device else "В исследовательских целях"

    @property
    def image_warning(self) -> str:  # п. 3.4 — формулировка отличается от SR
        return "Для поддержки принятия решений" if self.registered_medical_device else "В исследовательских целях"

    def additional_series_modality(self, original_modality: str) -> str:
        """Прил. 7: для ПУМ — строго ASMT, для ПУМ.МОСМЕДИИ — модальность оригинального исследования."""
        return "ASMT" if self.integration == "pum" else (original_modality or "OT")


DEFAULT = ServiceConfig()


def series_uid_by_mask(original_series_uid: str, model_id: int, add_id: int = 1, count: int = 1) -> str:
    """Прил. 7: {OriginalSeriesUID}.{modelId}.{addId}.{count}, оригинальный UID обрезается до 56 символов,
    итог — не длиннее 64 символов."""
    suffix = f".{model_id}.{add_id}.{count}"
    base = original_series_uid[:56]
    base = base[: MAX_UID_LENGTH - len(suffix)].rstrip(".")
    return base + suffix
