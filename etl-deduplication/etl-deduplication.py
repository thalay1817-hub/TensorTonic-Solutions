def deduplicate(records, key_columns, strategy="first"):
    
    result = {}
    order = []

    for rec in records:
        key = tuple(rec[col] for col in key_columns)

        if key not in result:
            result[key] = rec
            order.append(key)
        else:
            if strategy == "last":
                result[key] = rec

            elif strategy == "most_complete":
                current = result[key]

                # count None values
                current_none = sum(v is None for v in current.values())
                new_none = sum(v is None for v in rec.values())

                if new_none < current_none:
                    result[key] = rec

            # strategy "first" → do nothing

    return [result[k] for k in order]