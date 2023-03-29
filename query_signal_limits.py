import json
import argparse
from datetime import datetime


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help="date as 'Y-m-d H:M:S'", default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    args = parser.parse_args()

    # Load the JSON file as a list of dictionaries
    with open("data/output/dynamic_limits.json", 'r') as f:
        data = [json.loads(line) for line in f]

    # Convert the time strings to datetime objects
    for item in data:
        item["time"] = datetime.strptime(item["time"], "%Y-%m-%d %H:%M:%S")

    # Sort the data by time in descending order
    data.sort(key=lambda x: x["time"], reverse=True)

    # Find the closest past item
    for item in data:
        if item["time"] <= datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S"):
            closest_item = item
            break

    # Print the closest past item
    print(closest_item)