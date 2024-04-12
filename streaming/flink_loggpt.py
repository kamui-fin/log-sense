import json
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaOffsetsInitializer,
    KafkaSink,
    KafkaRecordSerializationSchema,
    DeliveryGuarantee,
)
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
from pyflink.datastream import (
    StreamExecutionEnvironment,
    ProcessWindowFunction,
    AggregateFunction,
)
from pyflink.datastream.window import TumblingEventTimeWindows, TimeWindow
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner, TemplateMinerConfig
from pyflink.datastream.state import (
    ValueStateDescriptor,
    StateTtlConfig,
    MapStateDescriptor,
    ListStateDescriptor,
)
from transformers import BertTokenizer
import re
import hashlib

from streaming.utils import JSONDeserializationSchema, LogTimestampAssigner, regex_clean

CONTEXT_SIZE = 5
num_special_tokens = 3
bos_token_id = 0
eos_token_id = 1
pad_token_id = 2


class LogAggregator(AggregateFunction):
    def __init__(self):
        # persistence = FilePersistence("drain3_state.bin")
        # config = TemplateMinerConfig()
        # config.profiling_enabled = True
        # self.template_miner = TemplateMiner(persistence_handler=persistence, config=config)
        self.template_miner = TemplateMiner()

    def create_accumulator(self):
        return {
            "chunks": [],
            "original_logs": [],
            "hashes": [],
            "current_tokens": {
                "input_ids": [],
            },
            "current_logs": [],
        }

    def add_current_chunk(self, accumulator):
        input_ids = accumulator["current_tokens"]["input_ids"]
        attention_mask = [1] * (len(input_ids) + 2) + [0] * (
            CONTEXT_SIZE - len(input_ids) - 2
        )
        pad_input_ids = (
            [bos_token_id]
            + input_ids
            + [eos_token_id]
            + [pad_token_id] * (CONTEXT_SIZE - len(input_ids) - 2)
        )
        accumulator["current_tokens"]["input_ids"] = pad_input_ids
        accumulator["current_tokens"]["attention_mask"] = attention_mask
        accumulator["current_tokens"]["labels"] = pad_input_ids

        accumulator["chunks"].append(accumulator["current_tokens"])
        accumulator["original_logs"].append(accumulator["current_logs"])
        sha256_hash = hashlib.sha256(
            "\n".join(accumulator["current_logs"]).encode("utf-8")
        ).hexdigest()
        accumulator["hashes"].append(
            str(int.from_bytes(sha256_hash[:8], byteorder="big"))
        )

    def drain_parse(self, log_line):
        return self.template_miner.add_log_message(log_line)["cluster_id"]

    def add(self, log_data, accumulator):
        clean_log = regex_clean(log_data["original_text"])
        sha256_hash = hashlib.sha256(clean_log.encode("utf-8")).digest()
        log_hash = str(int.from_bytes(sha256_hash[:8], byteorder="big"))
        template_id = self.drain_parse(clean_log) + num_special_tokens
        log_data = {"clean_log": clean_log, "hash": log_hash, **log_data}

        if (
            len(accumulator["current_tokens"]["input_ids"]) < CONTEXT_SIZE - 2
        ):  # -2 for bos and eos tokens
            accumulator["current_tokens"]["input_ids"].append(template_id)
            accumulator["current_logs"].append(log_data)
        else:
            self.add_current_chunk(accumulator)
            # Initialize new chunk
            accumulator["current_tokens"] = {"input_ids": [template_id]}
            accumulator["current_logs"] = [log_data]
        return accumulator

    def get_result(self, accumulator):
        return {
            "hashes": accumulator["hashes"],
            "chunks": accumulator["chunks"],
            "original_logs": accumulator["original_logs"],
        }

    def merge(self, acc1, acc2):
        # merging isn't necessary in this case
        print(f"MERGING {acc1} {acc2}")
        raise NotImplementedError()


env = StreamExecutionEnvironment.get_execution_environment()

watermark_strategy = (
    WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(1))
    .with_idleness(Duration.of_millis(500))
    .with_timestamp_assigner(LogTimestampAssigner())
)
configured_services = ["service1"]  # TODO: pull from global service list

for current_svc in configured_services:
    current_source = (
        KafkaSource.builder()
        .set_bootstrap_servers("localhost:9092")
        .set_topics(current_svc)
        .set_group_id("flink-group")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(JSONDeserializationSchema())
        .build()
    )
    current_stream = env.from_source(current_source, watermark_strategy, current_svc)
    windowed_stream = (
        current_stream.key_by(
            lambda log_data: log_data["node"], key_type=Types.STRING()
        )
        .window(TumblingEventTimeWindows.of(Time.seconds(30)))
        .aggregate(LogAggregator())
    )
    sink = (
        KafkaSink.builder()
        .set_bootstrap_servers("localhost:9092")
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(f"{current_svc}-gpt-processed")
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )
    windowed_stream.sink_to(sink)

env.execute()
