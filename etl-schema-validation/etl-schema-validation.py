def validate_records(records, schema):
    results = []
    
    for idx, record in enumerate(records):
        errors = []
        
        for col_def in schema:
            col = col_def["column"]
            expected_type = col_def["type"]
            nullable = col_def.get("nullable", False)
            
            # 1. Missing column
            if col not in record:
                errors.append(f"{col}: missing")
                continue
            
            value = record[col]
            
            # 2. Null check
            if value is None:
                if not nullable:
                    errors.append(f"{col}: null")
                continue  # skip further checks
            
            # 3. Type check
            actual_type = type(value).__name__
            
            type_ok = False
            if expected_type == "int":
                type_ok = type(value) is int
            elif expected_type == "float":
                type_ok = type(value) in (int, float)
            elif expected_type == "str":
                type_ok = type(value) is str
            
            if not type_ok:
                errors.append(f"{col}: expected {expected_type}, got {actual_type}")
                continue  # skip range check
            
            # 4. Range check
            if expected_type in ("int", "float"):
                if "min" in col_def and value < col_def["min"]:
                    errors.append(f"{col}: out of range")
                    continue
                if "max" in col_def and value > col_def["max"]:
                    errors.append(f"{col}: out of range")
                    continue
        
        results.append((idx, len(errors) == 0, errors))
    
    return results