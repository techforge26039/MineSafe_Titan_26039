"""Fast dependency-light smoke test for the MineSafe Titan core."""
from minesafe.core.telemetry import demo_telemetry, sanitize_and_validate
from minesafe.core.hazard import evaluate_hazard, calculate_trpi
from minesafe.core.route import safest_route
from minesafe.models.vision import VisionEvent, EVENT_WEIGHTS
from minesafe.core.methods import o2_buffer_proxy_minutes, reference_rows


def main():
    t = sanitize_and_validate(demo_telemetry("Compound Disaster"), 1, "SIMULATED")
    assert t.is_valid
    score, primary, secondary, breaches, components = evaluate_hazard(t)
    assert score >= 80 and breaches
    trpi, parts = calculate_trpi(t, 132, 93, True, True)
    assert trpi >= 70 and "Fall flag (10%)" in parts
    route, cost = safest_route("BASE", "ZONE_C", hazard_nodes=["C"])
    assert route and cost > 0
    assert EVENT_WEIGHTS["FALL"] == 95.0
    VisionEvent("FALL", 0.9, "fallen_person", 95).to_dict()
    assert o2_buffer_proxy_minutes(1000, 20.0, 4, 0.25) == 10000.0
    assert any(r['Parameter']=='Oxygen' for r in reference_rows())
    print("MineSafe Titan core smoke test: PASS")


if __name__ == "__main__":
    main()
