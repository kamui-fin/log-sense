import json
import pathlib
import re
from pprint import pprint
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.formats.json import JsonRowDeserializationSchema
from pyflink.datastream.connectors.file_system import FileSource
from pyflink.common import Time, WatermarkStrategy, Duration
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext, FlatMapFunction
from pyflink.common import Types, WatermarkStrategy, Time, Encoder
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import (
    StreamExecutionEnvironment,
    ProcessWindowFunction,
    AggregateFunction,
)
from pyflink.datastream.window import TumblingEventTimeWindows, TimeWindow
from pyflink.datastream.state import (
    ValueStateDescriptor,
    StateTtlConfig,
    MapStateDescriptor,
    ListStateDescriptor,
)
from transformers import BertTokenizer
import re
import hashlib
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer, KafkaSink, KafkaRecordSerializationSchema, DeliveryGuarantee

class JSONSerializationSchema():
    def serialize(self, obj):
        return json.dumps(obj).encode('utf-8')

class JSONDeserializationSchema(SimpleStringSchema):
    def deserialize(self, message):
        return json.loads(message)

# currently only for HDFS rn
def regex_clean(log_line):
    blk_regex = re.compile("blk_-?\d+")
    ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
    num_regex = re.compile("\d*\d")

    log_line = re.sub(blk_regex, "BLK", log_line)
    log_line = re.sub(ip_regex, "IP", log_line)
    log_line = re.sub(num_regex, "NUM", log_line)

    return log_line

class LogAggregator(AggregateFunction):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def create_accumulator(self):
        return {"unique_logs": {}, "all_logs": []}

    def add(self, log, accumulator):
        unique_logs, all_logs = accumulator["unique_logs"], accumulator["all_logs"]
        clean_log = regex_clean(log["log"])
        hash_log = hashlib.sha1(clean_log.encode("utf-8")).hexdigest()
        tokens = self.tokenizer(
            clean_log, padding="max_length", truncation=True, max_length=512
        )
        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]

        if hash_log not in unique_logs:
            unique_logs[hash_log] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **log,
            }

        all_logs.append(hash_log)
        return accumulator

    def get_result(self, accumulator):
        return accumulator

    def merge(self, acc1, acc2):
        return NotImplementedError()


class LogTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, log_data, record_timestamp) -> int:
        return log_data["timestamp"]

env = StreamExecutionEnvironment.get_execution_environment()
env.add_jars("file:///home/kamui/dev/projects/log-sense/streaming/jar/flink-connector-kafka-3.0.2-1.18.jar", "file:///home/kamui/dev/projects/log-sense/streaming/jar/kafka-clients-3.7.0.jar")

configured_services = ['test']

watermark_strategy = (
    WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(1))
    .with_idleness(Duration.of_millis(500))
    .with_timestamp_assigner(LogTimestampAssigner())
)
for current_svc in configured_services:
    current_source = KafkaSource.builder() \
        .set_bootstrap_servers("localhost:9092") \
        .set_topics(current_svc) \
        .set_group_id("flink-group") \
        .set_starting_offsets(KafkaOffsetsInitializer.latest()) \
        .set_value_only_deserializer(JSONDeserializationSchema()) \
        .build()
    current_stream = env.from_source(current_source, watermark_strategy, current_svc)
    windowed_stream = (
        current_stream.map(lambda x: json.loads(x))
        .window_all(TumblingEventTimeWindows.of(Time.seconds(60)))
        .aggregate(LogAggregator())
        .map(lambda x: json.dumps(x), Types.STRING())
    )
    sink = KafkaSink.builder() \
        .set_bootstrap_servers("localhost:9092") \
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
                .set_topic(f"{current_svc}-processed")
                .set_value_serialization_schema(SimpleStringSchema())
                .build()
        ) \
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE) \
        .build()
    windowed_stream.sink_to(sink)

env.execute()