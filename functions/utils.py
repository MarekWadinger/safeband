def common_prefix(topics: list) -> str:
    """Find the common prefix among a list of topics and return it, including
    the last '/' character.

    Args:
        topics (list): A list of string topics.

    Returns:
        str: The common prefix among the topics, including the last '/'
        character.

    Examples:
    >>> topics = [
    ...     "topic/emeter/0/power",
    ...     "topic/emeter/1/power"]
    >>> common_prefix(topics)
    'topic/emeter/'

    >>> topics = [
    ...     "power",
    ...     "temp"]
    >>> common_prefix(topics)
    ''

    >>> topics = [
    ...     "plant_1/emeter/0/power",
    ...     "plant_2/emeter/0/power"]
    >>> common_prefix(topics)
    ''

    >>> topics = None
    >>> common_prefix(topics)
    ''
    """
    if not topics:
        return ""

    # Sort the list of topics to ensure the shortest and longest topics
    # are at the extremes
    topics.sort()

    # Take the first and last topics for comparison
    first_str = topics[0]
    last_str = topics[-1]

    # Find the minimum length between the first and last topics
    min_len = min(len(first_str), len(last_str))

    # Initialize a prefix variable to store the common prefix
    prefix = ""

    # Compare the characters in the first and last topics
    for i in range(min_len):
        if first_str[i] == last_str[i]:
            prefix += first_str[i]
        else:
            break

    # Find the last '/' character in the common prefix
    slash_index = prefix.rfind("/")

    # If a '/' character is found, return everything before it (including
    # the '/')
    if slash_index >= 0:
        return prefix[: slash_index + 1]

    return ""
