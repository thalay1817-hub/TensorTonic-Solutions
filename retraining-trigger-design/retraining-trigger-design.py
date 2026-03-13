def retraining_policy(daily_stats, config):
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    budget = config["budget"]

    retrain_days = []

    days_since_retrain = 0
    last_retrain_day = -cooldown  # ensures cooldown satisfied on day 1

    for stat in daily_stats:
        day = stat["day"]
        drift = stat["drift_score"]
        performance = stat["performance"]

        days_since_retrain += 1

        # Trigger conditions
        drift_trigger = drift > drift_threshold
        performance_trigger = performance < performance_threshold
        staleness_trigger = days_since_retrain >= max_staleness

        trigger = drift_trigger or performance_trigger or staleness_trigger

        # Constraints
        cooldown_ok = (day - last_retrain_day) >= cooldown
        budget_ok = budget >= retrain_cost

        if trigger and cooldown_ok and budget_ok:
            retrain_days.append(day)
            budget -= retrain_cost
            last_retrain_day = day
            days_since_retrain = 0

    return retrain_days


# Example 1
daily_stats = [
   {"day": 1, "drift_score": 0.1, "performance": 0.95},
   {"day": 2, "drift_score": 0.3, "performance": 0.93},
   {"day": 3, "drift_score": 0.6, "performance": 0.90},
   {"day": 4, "drift_score": 0.2, "performance": 0.94},
]

config = {
    "drift_threshold": 0.5,
    "performance_threshold": 0.7,
    "max_staleness": 30,
    "cooldown": 1,
    "retrain_cost": 100,
    "budget": 500
}

print(retraining_policy(daily_stats, config))


# Example 2
daily_stats = [
   {"day": 1, "drift_score": 0.8, "performance": 0.90},
   {"day": 2, "drift_score": 0.8, "performance": 0.90},
   {"day": 3, "drift_score": 0.8, "performance": 0.90},
   {"day": 4, "drift_score": 0.8, "performance": 0.90},
]

config = {
    "drift_threshold": 0.5,
    "performance_threshold": 0.7,
    "max_staleness": 30,
    "cooldown": 3,
    "retrain_cost": 100,
    "budget": 500
}

print(retraining_policy(daily_stats, config))