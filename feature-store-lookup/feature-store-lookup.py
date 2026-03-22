def feature_store_lookup(feature_store, requests, defaults):
    result = []
    
    for req in requests:
        user_id = req.get("user_id")
        online_features = req.get("online_features", {})
        
        # Get offline features or use defaults
        offline_features = feature_store.get(user_id, defaults)
        
        # Merge offline and online features
        combined = {**offline_features, **online_features}
        
        result.append(combined)
    
    return result