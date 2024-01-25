from datetime import datetime

import pytz


def now():
    return datetime.now().astimezone(pytz.utc).timestamp()

