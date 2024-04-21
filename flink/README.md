# LogSense data-processing stage

## Overview

1. Pipe Loki tail websocket streams into Kafka topics. Each service gets its own topic. A log in a kafka topic for HDFS might look like:

```json
{
    "block": "blk_892839",
    "timestamp": 90392804,
    "file": "hdfs_master.log",
    "log": "[INFO] 2023-01-23 Unable to update block blk_89839..."
}
```

2. Flink consume JSON from Kafka and deserialize timestamps, metadata, and other fields.

3. Regex clean

4. Tokenize

5. Group by node/block

6. Time fixed windows of 1 minute

7. Further chunk each time window into 512 token chunks. 

8. Assign timestamp range to each chunk

9. Build array of unique log strings across the chunks.

10. Map each log index of OG stream to unique log index.

11. Send off processed logs into kafka sink. Example output:

```json

{
    "topic_destination": "HDFS",
    "payload": [
        { 
            "unique_logs": { 
                0x02342: [2, 3, 0, 1, 3]
                },
            "chunks": [
                {
                    "unique_log_id": 0x02342,
                    "timestamp_start": 829347892,
                    "timestamp_end": 82957892,
                    "files": ["hdfs_main_sys.log", "hdfs_main_net.log"]
                }
            ]
        }
     ]
},
```