from pathlib import Path
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
import re
from pyflink.datastream.functions import MapFunction
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaOffsetsInitializer,
    KafkaSink,
    KafkaRecordSerializationSchema,
    DeliveryGuarantee,
)
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Time, WatermarkStrategy, Duration
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext
from pyflink.common import Types, WatermarkStrategy, Time
from pyflink.datastream import (
    StreamExecutionEnvironment,
    AggregateFunction,
)
from pyflink.datastream.window import TumblingEventTimeWindows
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from pyflink.datastream.state import (
    ValueStateDescriptor,
)
from transformers import BertTokenizer
import requests
import hashlib
from models import GlobalConfig, ServiceConfig
from utils import JSONDeserializationSchema, LogTimestampAssigner, regex_clean

num_special_tokens = 3
bos_token_id = 0
eos_token_id = 1
pad_token_id = 2

LOGSENSE_BACKEND_URI = os.getenv("LOGSENSE_BACKEND_URI", "http://localhost:3000")
KAFKA_URI = os.getenv("KAFKA_URI", "localhost:9092")


class LogAggregator(AggregateFunction):
    def __init__(self):
        persistence = FilePersistence("/tmp/drain3_state.bin")
        self.template_miner = TemplateMiner(persistence_handler=persistence)

        self.context_size = None

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
            self.context_size - len(input_ids) - 2
        )
        pad_input_ids = (
            [bos_token_id]
            + input_ids
            + [eos_token_id]
            + [pad_token_id] * (self.context_size - len(input_ids) - 2)
        )
        accumulator["current_tokens"]["input_ids"] = pad_input_ids
        accumulator["current_tokens"]["attention_mask"] = attention_mask
        accumulator["current_tokens"]["labels"] = pad_input_ids

        accumulator["chunks"].append(accumulator["current_tokens"])
        accumulator["original_logs"].append(accumulator["current_logs"])
        sha256_hash = hashlib.sha256(
            "\n".join(
                [log["cleaned_text"] for log in accumulator["current_logs"]]
            ).encode("utf-8")
        ).digest()
        accumulator["hashes"].append(
            str(int.from_bytes(sha256_hash[:8], byteorder="big"))
        )

    def drain_parse(self, log_line):
        return self.template_miner.add_log_message(log_line)["cluster_id"]

    def add(self, log_data, accumulator):
        if self.context_size is None:
            self.context_size = config.configs[log_data["service"]].context_size

        clean_log = regex_clean(log_data["original_text"])
        sha256_hash = hashlib.sha256(clean_log.encode("utf-8")).digest()
        log_hash = str(int.from_bytes(sha256_hash[:8], byteorder="big"))
        template_id = self.drain_parse(clean_log) + num_special_tokens
        log_data = {"cleaned_text": clean_log, "hash": log_hash, **log_data}

        if (
            len(accumulator["current_tokens"]["input_ids"]) < self.context_size - 2
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
        print("FINISHED RESULT: get_result()")
        if accumulator["current_logs"]:
            self.add_current_chunk(accumulator)
        return {
            "hashes": accumulator["hashes"],
            "chunks": accumulator["chunks"],
            "original_logs": accumulator["original_logs"],
        }

    def merge(self, acc1, acc2):
        # merging isn't necessary in this case
        raise NotImplementedError()


def get_config() -> GlobalConfig:
    response = requests.get(
        f"{LOGSENSE_BACKEND_URI}/api/trpc/config.getServices"
    ).json()
    service_list = response["result"]["data"]["json"]["data"]
    configs = {}
    for service in service_list:
        name = service["name"]
        del service["_id"]
        del service["name"]
        del service["description"]
        del service["__v"]
        configs[name] = ServiceConfig(**service)
    return GlobalConfig(configs=configs)


config = get_config()


class TrainModeProcessor(MapFunction):
    def __init__(self):
        super().__init__()
        self.counter_state = None
        self.train_mode = None

    def open(self, runtime_context: RuntimeContext):
        # Initialize counter state
        self.counter_state = runtime_context.get_state(
            ValueStateDescriptor("counter", Types.INT())
        )
        self.train_mode = runtime_context.get_state(
            ValueStateDescriptor("train_mode", Types.BOOLEAN())
        )

    def map(self, value):
        service = value["service"]
        self.train_mode.update(config.configs[service].is_train)
        if self.counter_state.value() is None:
            self.counter_state.update(0)

        if self.train_mode.value():
            self.counter_state.update(self.counter_state.value() + 1)
            return {
                "train_strategy": (
                    "pre-train"
                    if self.counter_state.value() - 1
                    <= config.configs[service].max_pretrain
                    else "finetune"
                ),
                **value,
            }
        # Emit result
        return value


class TraceExtractor(MapFunction):
    def __init__(self):
        super().__init__()

    def map(self, value):
        svc_config = config.configs[value["service"]]
        if svc_config.enable_trace:
            trace_regex = re.compile(svc_config.trace_regex)
            trace_id = trace_regex.findall(value["original_text"])
            if len(trace_id):
                return {
                    "trace_id": trace_id[0],
                    **value,
                }

        return value


def trace_node_selector(value):
    svc_config = config.configs[value["service"]]
    if svc_config.enable_trace:
        return value["trace_id"]
    else:
        return value["node"]


class RegexTokenize(MapFunction):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def map(self, log_data):
        clean_log = regex_clean(log_data["original_text"])
        sha256_hash = hashlib.sha256(clean_log.encode("utf-8")).digest()
        hash_log = str(int.from_bytes(sha256_hash[:8], byteorder="big"))
        tokens = self.tokenizer(
            clean_log, padding="max_length", truncation=True, max_length=512
        )
        processed_log = {
            "cleaned_text": clean_log,
            "hash": hash_log,
            "tokens": dict(tokens),
            **log_data,
        }
        return processed_log


env = StreamExecutionEnvironment.get_execution_environment()
current_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
env.add_jars(
    "file://" + str(current_file_path / "jar" / "flink-connector-kafka-3.0.2-1.18.jar"),
    "file://" + str(current_file_path / "jar" / "kafka-clients-3.7.0.jar"),
)

watermark_strategy = (
    WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(1))
    .with_idleness(Duration.of_millis(500))
    .with_timestamp_assigner(LogTimestampAssigner())
)

source = (
    KafkaSource.builder()
    .set_bootstrap_servers(KAFKA_URI)
    .set_topics("loki")
    .set_group_id("flink")
    .set_starting_offsets(KafkaOffsetsInitializer.latest())
    .set_value_only_deserializer(JSONDeserializationSchema())
    .build()
)
kafka_stream = env.from_source(source, watermark_strategy, "flink-loggpt")


rapid_stream = (
    kafka_stream.map(lambda x: json.loads(x))
    .map(RegexTokenize())
    .map(lambda x: json.dumps(x), Types.STRING())
)
rapid_sink = (
    KafkaSink.builder()
    .set_bootstrap_servers(KAFKA_URI)
    .set_record_serializer(
        KafkaRecordSerializationSchema.builder()
        .set_topic(f"rapid-processed")
        .set_value_serialization_schema(SimpleStringSchema())
        .build()
    )
    .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
    .build()
)
rapid_stream.sink_to(rapid_sink)

# get tumbling window seconds from argparse
args = sys.argv
if len(args) < 2:
    window_size_sec = 60
else:
    window_size_sec = int(args[1])

log_gpt_stream = (
    kafka_stream.map(lambda x: json.loads(x))
    .key_by(lambda log_data: log_data["service"], key_type=Types.STRING())
    .map(TrainModeProcessor())
    .map(TraceExtractor())
    .key_by(trace_node_selector, key_type=Types.STRING())
    .window(TumblingEventTimeWindows.of(Time.seconds(window_size_sec)))
    .aggregate(LogAggregator())
    .map(lambda x: json.dumps(x), Types.STRING())
)
log_gpt_sink = (
    KafkaSink.builder()
    .set_bootstrap_servers(KAFKA_URI)
    .set_record_serializer(
        KafkaRecordSerializationSchema.builder()
        .set_topic(f"gpt-processed")
        .set_value_serialization_schema(SimpleStringSchema())
        .build()
    )
    .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
    .build()
)
log_gpt_stream.sink_to(log_gpt_sink)

env.execute()
