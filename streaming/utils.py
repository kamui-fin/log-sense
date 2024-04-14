import re
import json 
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.common.serialization import SimpleStringSchema

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
def regex_clean(line):
    # blk_regex = re.compile("blk_-?\d+")
    # ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
    # num_regex = re.compile("\d*\d")

    # log_line = re.sub(blk_regex, "BLK", log_line)
    # log_line = re.sub(ip_regex, "IP", log_line)
    # log_line = re.sub(num_regex, "NUM", log_line)

    is_anomaly_regex = re.compile("^- ")
    date_time_regex = re.compile(
        "\d{1,4}\-\d{1,2}\-\d{1,2}-\d{1,2}.\d{1,2}.\d{1,2}.\d{1,6}"
    )
    date_regex = re.compile("\d{1,4}\.\d{1,2}\.\d{1,2}")
    ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
    server_regex = re.compile("\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[:]+)\S+")
    server_regex2 = re.compile("\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[-])\S+")
    ecid_regex = re.compile("[A-Z0-9]{28}")
    serial_regex = re.compile("[a-zA-Z0-9]{48}")
    memory_regex = re.compile("0[xX][0-9a-fA-F]\S+")
    path_regex = re.compile(".\S+(?=.[0-9a-zA-Z])(?=[/]).\S+")
    iar_regex = re.compile("[0-9a-fA-F]{8}")
    num_regex = re.compile("(\d+)")

    line = re.sub(is_anomaly_regex, "", line)
    line = re.sub(date_time_regex, "DT", line)
    line = re.sub(date_regex, "DATE", line)
    line = re.sub(ip_regex, "IP", line)
    line = re.sub(server_regex, "NODE", line)
    line = re.sub(server_regex2, "NODE", line)
    line = re.sub(ecid_regex, "ECID", line)
    line = re.sub(serial_regex, "SERIAL", line)
    line = re.sub(memory_regex, "MEM", line)
    line = re.sub(path_regex, "PATH", line)
    line = re.sub(iar_regex, "IAR", line)
    line = re.sub(num_regex, "NUM", line)

    return line
