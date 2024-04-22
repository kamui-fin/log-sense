import re
import json
from typing import List
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.common.serialization import SimpleStringSchema

from models import RegexSub


class LogTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, log_data, record_timestamp) -> int:
        return log_data["timestamp"]


class JSONSerializationSchema:
    def serialize(self, obj):
        return json.dumps(obj).encode("utf-8")


class JSONDeserializationSchema(SimpleStringSchema):
    def deserialize(self, message):
        return json.loads(message)


# currently only hardcoded
def regex_clean(line, regex_subs: List[RegexSub]):
    for regex_sub in regex_subs:
        line = re.sub(regex_sub.pattern, regex_sub.replacement, line)
    return line