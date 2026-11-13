"""Calorie lookup and meal summary for detected dishes.

Kept separate from the Streamlit UI so it can be unit-tested without a model or a browser.
Calories are approximate values for one standard serving / piece of each dish.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# kcal per standard serving (1 piece for roti, idli, samosa, jalebi, modak, gulab jamun, bhatura, dhokla)
KCAL_PER_SERVING: dict[str, int] = {
    "Bhatura": 300, "BhindiMasala": 150, "Biryani": 350, "Chole": 250,
    "ShahiPaneer": 380, "chicken": 250, "dal": 150, "dhokla": 60,
    "gulab_jamun": 150, "idli": 50, "jalebi": 150, "modak": 150,
    "palak_paneer": 280, "poha": 250, "rice": 130, "roti": 85, "samosa": 150,
}

DISPLAY_NAME: dict[str, str] = {
    "BhindiMasala": "Bhindi Masala", "ShahiPaneer": "Shahi Paneer", "chicken": "Chicken curry",
    "dal": "Dal", "dhokla": "Dhokla", "gulab_jamun": "Gulab Jamun", "idli": "Idli", "jalebi": "Jalebi",
    "modak": "Modak", "palak_paneer": "Palak Paneer", "poha": "Poha", "rice": "Rice", "roti": "Roti",
    "samosa": "Samosa",
}

HIGH_CALORIE_MEAL = 800  # kcal; above this the app suggests portion control


@dataclass(frozen=True)
class Line:
    label: str
    name: str
    count: int
    kcal_each: int

    @property
    def kcal(self) -> int:
        return self.count * self.kcal_each


@dataclass(frozen=True)
class MealSummary:
    lines: list[Line]

    @property
    def total_kcal(self) -> int:
        return sum(line.kcal for line in self.lines)

    @property
    def is_high_calorie(self) -> bool:
        return self.total_kcal > HIGH_CALORIE_MEAL


def display_name(label: str) -> str:
    return DISPLAY_NAME.get(label, label)


def summarize(labels: list[str]) -> MealSummary:
    """Turn the list of detected class labels (one per box) into per-dish counts and calories.

    Unknown labels are kept with 0 kcal so they still show up instead of silently disappearing.
    Lines are sorted by calories, highest first.
    """
    counts = Counter(labels)
    lines = [Line(label=lb, name=display_name(lb), count=n, kcal_each=KCAL_PER_SERVING.get(lb, 0)) for lb, n in counts.items()]
    lines.sort(key=lambda line: (-line.kcal, line.name))
    return MealSummary(lines=lines)
