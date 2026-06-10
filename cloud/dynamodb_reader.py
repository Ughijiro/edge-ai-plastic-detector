import boto3
from decimal import Decimal

AWS_REGION = "eu-north-1"
TABLE_NAME = "PlasticDetectorEvents"


def decimal_to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: decimal_to_float(val) for key, val in value.items()}
    if isinstance(value, list):
        return [decimal_to_float(item) for item in value]
    return value


def get_all_events():
    """
    Reads all events from DynamoDB.

    This is acceptable for the demo because the table contains only a small
    number of events. For a larger system, pagination and date filtering
    should be added.
    """
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(TABLE_NAME)

    response = table.scan()
    items = response.get("Items", [])

    events = [decimal_to_float(item) for item in items]

    events.sort(key=lambda event: event.get("timestamp", ""), reverse=True)

    return events