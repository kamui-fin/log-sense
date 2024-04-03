# Grouping mechanism analysis

In this document, I'd like to analyze some of the differences in grouping mechanisms across datasets and models used (RAPID & LogGPT).

## LogGPT

Within the logs for an HDFS node, we concatenate all the event IDs, which is a number used to represent the log template of a specific log. To add an upper bound for inferencing, we limit this with a tumbling window of 1 minute. with additional chunking.

For other datasets like BGL (single-node), we skip the grouping due to lack of node IDs, and then apply the same windowing. 

However, for our more general framework that's being applied to distributed systems, we believe it's better to still include the grouping. So in general, for LogGPT, the pre-processing looks like:

1. Group by node
2. Window ~1 min
3. Log parse into event IDs + `num_special_tokens`
4. Further chunking to cap at 512 tokens per batch sample.
4. Encode the timestamp range + unique file sources to identify the log in Grafana

Output: 

```json
[
    {
        "node": "",
        "tokens": [],
        "attention_mask": [],
        "timestamp_start": 0,
        "timestamp_end": 0,
        "unique_files": []
    },
    ...
]
```

### Hadoop

In our case study, we focus on applying it to a Hadoop MapReduce cluster. Previously, a chunk within the 1m time window for a node/block would be sent for inference. Instead of a node or block here, it's a specific job. 


## RAPID

TBA..

## Generaling into a framework

As you can see, different applications require different grouping styles. That part would be need to be configured within LogSense. 