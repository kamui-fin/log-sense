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
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
from pyflink.common import Types, WatermarkStrategy, Time, Encoder
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import StreamExecutionEnvironment, ProcessWindowFunction, AggregateFunction
from pyflink.datastream.window import TumblingEventTimeWindows, TimeWindow
from pyflink.datastream.state import (
    ValueStateDescriptor,
    StateTtlConfig,
    MapStateDescriptor,
    ListStateDescriptor,
)
from processing import *
from transformers import BertTokenizer
import re

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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def create_accumulator(self):
        return {
            "chunks": [],
            "current_chunk": {
                "num_tokens": 0,
                "input_ids": [],
                "attention_mask": [],
                "concatenated_text": "",
                "start_timestamp": 0,
                "end_timestamp": 0,
                "file_sources": set(),
            },
        }
    
    def add(self, log_data, accumulator):
        clean_log = regex_clean(log_data['log'])
        tokens = self.tokenizer(clean_log, padding=False, truncation=True, max_length=512)
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        starting_accum = accumulator['current_chunk']['num_tokens'] == 0
        if not starting_accum:
            input_ids, attention_mask = input_ids[1:], attention_mask[1:]
        else:
            accumulator['current_chunk']['start_timestamp'] = log_data['timestamp']
        accumulator['current_chunk']['num_tokens'] += len(input_ids)
        accumulator['current_chunk']['end_timestamp'] = log_data['timestamp']
        if accumulator['current_chunk']['num_tokens'] <= 512:
            accumulator['current_chunk']['input_ids'].extend(input_ids)
            accumulator['current_chunk']['attention_mask'].extend(attention_mask)
            accumulator['current_chunk']['concatenated_text'] += clean_log
            accumulator['current_chunk']['file_sources'].add(log_data['filename'])
        else:
            pad_input_ids = accumulator['current_chunk']['input_ids'] + [0] * (512 - len(accumulator['current_chunk']['input_ids']))
            pad_mask = accumulator['current_chunk']['attention_mask'] + [0] * (512 - len(accumulator['current_chunk']['attention_mask']))
            accumulator['current_chunk']['input_ids'] = pad_input_ids
            accumulator['current_chunk']['attention_mask'] = pad_mask
            accumulator['chunks'].append(accumulator['current_chunk'])
            # Initialize new chunk
            accumulator['current_chunk'] = {
                "num_tokens": len(input_ids),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "concatenated_text": clean_log,
                "start_timestamp": log_data['timestamp'],
                "end_timestamp": log_data['timestamp'],
                "file_sources": set([log_data['filename']])
            }
        return accumulator

    def get_result(self, accumulator):
        pad_input_ids = accumulator['current_chunk']['input_ids'] + [0] * (512 - len(accumulator['current_chunk']['input_ids']))
        pad_mask = accumulator['current_chunk']['attention_mask'] + [0] * (512 - len(accumulator['current_chunk']['attention_mask']))
        accumulator['current_chunk']['input_ids'] = pad_input_ids
        accumulator['current_chunk']['attention_mask'] = pad_mask
        accumulator['chunks'].append(accumulator['current_chunk'])
        return accumulator['chunks']

    def merge(self, acc1, acc2):
        # merging isn't necessary in this case
        print(f"MERGING {acc1} {acc2}")
        raise NotImplementedError()


class MinuteWindowProcessor(ProcessWindowFunction):
    def process(self,
        key: str,
        context: ProcessWindowFunction.Context,
        elements: Iterable[Dict[str, List[int]]]):
        for chunk in elements:
            yield chunk

class LogTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, log_data, record_timestamp) -> int:
        return log_data["timestamp"]


env = StreamExecutionEnvironment.get_execution_environment()

# dummy testing data
json_data = json.loads(pathlib.Path("./data/kafka_dummy.json").read_text())
for log in json_data:
    log['timestamp'] = int(str(log['timestamp'])[:14])
collection = sorted(json_data, key=lambda x: x['timestamp'])

data_stream = env.from_collection(collection=collection)

watermark_strategy = (
    WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(1))
    .with_idleness(Duration.of_millis(500))
    .with_timestamp_assigner(LogTimestampAssigner())
)
data_stream = data_stream.assign_timestamps_and_watermarks(watermark_strategy)

windowed_stream = (
    data_stream.key_by(lambda log_data: log_data["block"], key_type=Types.STRING())
    .window(TumblingEventTimeWindows.of(Time.minutes(1)))
    .aggregate(LogAggregator(), window_function=MinuteWindowProcessor())
    .print()
)

env.set_parallelism(1)
env.execute()  # submit job for execution
