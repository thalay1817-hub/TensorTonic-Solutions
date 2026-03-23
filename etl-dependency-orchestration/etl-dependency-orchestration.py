def schedule_pipeline(tasks, resource_budget):
    # Build task lookup
    task_map = {t["name"]: t for t in tasks}
    
    # Track state
    completed = set()
    running = []  # (end_time, name, resources)
    started = set()
    schedule = []
    
    time = 0
    
    while len(completed) < len(tasks):
        # 1. Complete finished tasks
        new_running = []
        for end_time, name, res in running:
            if end_time <= time:
                completed.add(name)
            else:
                new_running.append((end_time, name, res))
        running = new_running
        
        # 2. Find ready tasks
        ready = []
        for t in tasks:
            name = t["name"]
            if name in started:
                continue
            if all(dep in completed for dep in t["depends_on"]):
                ready.append(name)
        
        # Sort alphabetically
        ready.sort()
        
        # Current resource usage
        used_resources = sum(res for _, _, res in running)
        
        # 3. Schedule tasks greedily
        for name in ready:
            t = task_map[name]
            res = t["resources"]
            
            if used_resources + res <= resource_budget:
                start_time = time
                end_time = time + t["duration"]
                
                running.append((end_time, name, res))
                started.add(name)
                schedule.append((name, start_time))
                
                used_resources += res
        
        # 4. Advance time to next completion
        if running:
            time = min(end_time for end_time, _, _ in running)
        else:
            break  # no progress possible
    
    # Sort result
    schedule.sort(key=lambda x: (x[1], x[0]))
    
    return schedule