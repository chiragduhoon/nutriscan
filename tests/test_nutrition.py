from nutrition import HIGH_CALORIE_MEAL, KCAL_PER_SERVING, summarize


def test_counts_and_totals():
    meal = summarize(["roti", "roti", "dal", "rice"])
    by_label = {line.label: line for line in meal.lines}
    assert by_label["roti"].count == 2 and by_label["roti"].kcal == 2 * KCAL_PER_SERVING["roti"]
    assert meal.total_kcal == 2 * 85 + 150 + 130


def test_empty_plate():
    meal = summarize([])
    assert meal.lines == [] and meal.total_kcal == 0 and not meal.is_high_calorie


def test_unknown_label_is_kept_with_zero_kcal():
    meal = summarize(["pizza", "idli"])
    assert {line.label for line in meal.lines} == {"pizza", "idli"}
    assert meal.total_kcal == KCAL_PER_SERVING["idli"]


def test_sorted_by_calories_and_display_names():
    meal = summarize(["idli", "Biryani", "gulab_jamun"])
    assert [line.name for line in meal.lines] == ["Biryani", "Gulab Jamun", "Idli"]


def test_high_calorie_threshold():
    assert not summarize(["Biryani"] * 2).is_high_calorie          # 700
    assert summarize(["Biryani"] * 3).is_high_calorie              # 1050
    assert HIGH_CALORIE_MEAL == 800
