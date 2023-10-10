from river import compose


def convert_to_nested_dict(d):
    """Convert flat parameters dict to nested dict.

    Examples:
    >>> input_dict = {
    ...     'QuantileFilter__a': 0.95,
    ...     'QuantileFilter__b': 0.95,
    ...     'Scaler__a': 1}
    >>> convert_to_nested_dict(input_dict)
    {'QuantileFilter': {'a': 0.95, 'b': 0.95}, 'Scaler': {'a': 1}}

    >>> input_dict = {
    ...     'QuantileFilter__a__round': 0.95,
    ...     'QuantileFilter__b__int': 0.95}
    >>> convert_to_nested_dict(input_dict)
    {'QuantileFilter': {'a': 1, 'b': 0}}
    """
    result: dict = {}
    for key, value in d.items():
        parts = key.split('__')
        if len(parts) <= 2:
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        elif len(parts) == 3:
            current = result
            for part in parts[:-2]:
                current = current.setdefault(part, {})
                type_ = eval(parts[-1])
                if isinstance(value, list):
                    value = [type_(v) for v in value]
                else:
                    value = type_(value)
            current[parts[-2]] = value
        else:
            raise ValueError(f"Up to 3 parts supported. You gave {len(parts)}")
    return result


def init_step(step, params):
    return step(**params.get(step.__name__, {}))


def nest_step(steps, params):
    if not isinstance(steps, list):
        steps = [steps]
    if len(steps) == 1:
        return init_step(steps[0], params)
    else:
        first_step = steps[0]
        remaining_steps = steps[1::].copy()[0]
        nested_result = nest_step(remaining_steps, params)
        return first_step(nested_result, **params.get(first_step.__name__, {}))


def build_model(steps: list, params: dict):
    """Build river model from list of cls and parameters.

    Examples:
    >>> from river import anomaly, preprocessing
    >>> steps = [preprocessing.StandardScaler,
    ...     [anomaly.QuantileFilter, anomaly.OneClassSVM]]
    >>> input_dict = {
    ...     'QuantileFilter': {'q': 0.95},
    ...     'OneClassSVM': {'nu': 0.123}}
    >>> model = build_model(steps, input_dict)
    >>> model["QuantileFilter"].q
    0.95
    >>> model["QuantileFilter"].anomaly_detector.nu
    0.123
    """
    model = compose.Pipeline()
    for step in steps:
        if not isinstance(step, list):
            model |= init_step(step, params)
        else:
            model |= nest_step(step, params)
    if len(model.steps) == 1:
        model = model[list(model.steps.keys())[0]]
    return model


if __name__ == '__main__':
    import doctest
    doctest.testmod()
