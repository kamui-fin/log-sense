# LogSense

LogSense leverages an ensemble of state-of-the-art LLM-based log anomaly detection models in streamlining an architecture capable of real-time anomaly detection. Learn more through our [whitepaper](http://clovlog.com/logsense.pdf).

![](./Distributed%20Architecture.png)

## Installation

LogSense's development environment can be started with Docker Compose:

```bash
docker-compose up -d
```

Quick notes:

-   If running without Cuda, remove the GPU-related sections from `docker-compose.yml`.
-   Add in the minio credentials for the `train` micro-service.

## Testing

The `docker-compose.yml` file mainly serves as a way to test out LogSense locally. Given the example loki & promtail config files, you can stream in log lines into `./test/input_logs/service1.log`, serving as the input to the entire system.

Be sure to configure `service1` and `service2` in the frontend at [`http://localhost:3000`](http://localhost:3000).

## Production

Due to the variability in use cases, LogSense offers a highly flexible architecture that can be tailored to one's own needs. In particular, we advise users to create their own Kubernetes configurations for production that suite their requirements.

Aditionally, we invite users to contribute to our repository and reach out to us about their own use cases. We wish to build an open-source community centered around deploying production log-anomaly detection systems and hope that LogSense can be the starting point.

Feel free to submit any bug reports, pull requests, or potential ideas!

## Backlog

-   Model explanability through anomalous token visualization. The idea is we highlight tokens that caused the model to classify it as anomalous by creating a new metrics based on top-k and max-sim ideas.
-   Dynamic thresholding algorithm for RAPID.
-   Further expansion of dashboard.

## Condensed Project Report

LogSense can be broken down into 3 parts:

1. Stream Processing
2. Real-time Inferencing
3. HITL (Human-in-the-loop) Updating

In an ideal world, humans would go through logs as they come in and quickly respond to anomalies they detect. It would be like standing on a shore and picking up any sea shells (anomalies) as the waves of logs come at you. Unfortunately, current system logs resemble less of a peaceful wave brushing your feet and more of a relentless tsunami mercilessly crashing into you. Dealing with this tsunami requires effective use of our more powerful weapon -- computing.

Traditionally, algorithms such as PCA, Decision Trees, and SVMs [3] defined boundaries or. common patterns the logs had to follow, but the emergence of language encoders shifted the discussion away from traditional rule definition towards NLP. Specifically, the effectiveness of BERT and GPT prompted log anomaly detection algorithms to leverage their power in encoding a log line.

We selected 2 models as the focus of our study:

1. RAPID - Retrieval-oriented (train-free) [2]
2. LogGPT - GPT based with reinforcement learning [3]

Subsequently, two schools of thought emerged when deciding how to encode logs -- tokenising and parsing [3]. Tokenising, which RAPID follows, is the process of regexing a log line and immediately encoding it. Regex preprocessing is the process of replacing any specifics in the log lines that are irrelevant to the inference process with general tokens such as NUM for numbers or IP for ip addresses. In contrast, LogGPT employs the parsing method. Given a sequence of log lines, LogGPT uses the Drain parser [5] to define unique log templates, also known as log keys, and assign each one to a log line in the sequence. This sequence of log keys is then encoded using a language model, such as GPT, in order to perform inference.

As the data progresses through the inference process, each model's distinctive treatment results in conclusions that are grounded in its unique reasoning. Namely, the treatment of semantics between the two models set them up to be the perfect pair.

## Semantics: Why these models form the perfect tandem

RAPID is fast and disregards semantics while LogGPT is slower and considers the semantic relations between logs. This is the one sentence short-and-sweet on how they're different, but let's explore why. As mentioned, RAPID encodes each log line individually, meaning the relations between logs is completely lost. If a new log's tokens differ too much from a known normal log that is closest to it in the embedding space, it's regarded as an anomaly -- a log's neighbors have no impact on it.

On the other side, LogGPT uses the previous keys in the log sequence to predict the the next key. If this next key isn't in the top k predicted keys, it's regarded as an anomaly. One can immediately notice that the act of considering the past tokens to predict the next is a textbook example of incorporating semantics -- the relations between logs in the sequence is kept alive.

Why even use RAPID when LogGPT can consider semantics better? Speed. Although RAPID doesn't consider semantics, it still holds its own against LogGPT when it comes to accuracy. Its shortcomings in accounting for semantics are accounted for by the speed at which it delivers results. Since RAPID doesn't need to parse the logs or predict the next key for every key, it's very rapid (ha!). And this is where the ensemble technique comes in.

The basic premise of the ensemble is as follows. When a new log is generated, RAPID immediately lets us know if it's anomalous or not. Once a group of logs have been generated in a time window, LogGPT uses its parsing and next-token prediction techniques to produce its own inference. This inference is either used to strengthen RAPID's decision or refute it. This way, users can first get quick responses that uses less information (RAPID) and get a stronger inference a little later (LogGPT). Speed first, accuracy next.

## Architecture Outline

### 1. Stream Processing

After a node is connected to LogSense, it can process the node's logs in a streaming manner. The logs are first aggregated in Loki to be queried for context later on. They are then sent to a Kafka task queue that is distributed according to service (HDFS, BGL, Thunderbird, etc). This task queue is connected to a Flink listener which treats the data in a streaming manner, computing the tokens for each log line and sending them to the inferencing cluster. Throughout this process, the data is treated as a stream of water -- never stuck or stationary, always flowing forward.

### 2. Real-time Inferencing

Due to the immense amounts of data that needs to be inferenced, LogSense harnesses the power of horizontal scaling. Kubernetes is employed to open up new pods whenever necessary in order to serve the streaming data as efficiently as possible. Although the specifics regarding each model's inference algorithm isn't discussed here, we highly recommend the reader to read the original papers to gain a deeper understanding of the models [2][3]. The ensemble of the models is achieved by first sending RAPID's conclusion to the front-end to be displayed, while LogGPT is running in the background to produce its own conclusion. Once it finishes, it is also sent to the front-end, where the user confirms or denies additional anomalies LogGPT finds. Apache Kafka has the ability to orchestrate this intricate symphony, exploiting its distributed task queueing capabilities to efficiently handle the movement of the data throughout the architecture.

### 3. HITL (Human-in-the-loop) Updating

First off, its worth mentioning that to provide the models with initial information regarding the structure of normal logs, LogSense offers a Train Mode. When activated, train mode lets LogSense treat every generated log as normal, letting it seed the database of normal logs. These normal logs allow us to pretrain LogGPT and provide the initial data for RAPID to get a core set.

Once displayed in the front end, users have the ability to refute any anomalies that they know are normal. If a displayed log is marked as normal, a service is triggered to delete that log from the anomaly database and treat it as a normal log for any future inferencing. This updating technique incorporates human feedback by allowing users to detect any false positives -- strengthening the accuracy of the architecture as a whole. The database of normal logs we conglomerate can be used to fine tune LogGPT every 24 hours, rendering a better model everyday.

## Applications within Distributed Systems

LogSense offers a significant advantage in its seamless integration with distributed systems. By effortlessly aggregating data from multiple configured nodes, LogSense becomes an invaluable tool within distributed system environments. One of the key challenges in distributed systems is the cumbersome task of monitoring logs across numerous nodes. LogSense effectively addresses this challenge by aggregating logs across all nodes of a service, streamlining the log tracking process. The stream processing capabilities of Apache Flink fuels LogSense's ability to process logs at scale. The distributed properties of Flink leads to efficient processing of large amounts of logs as they occur, meaning users can respond to any anomalies within seconds of their occurrence. Once configured with LogSense, each node can be identified by a metadata tag specifying its service (such as HDFS, BGL, etc.), ensuring uniform treatment across all nodes within the service. LogSense can also "plug and play" into existing observability setups that utilize Loki and Grafana by simply latching onto LogSense's Loki.

## Monitoring

LogSense's front-end allows users to not only understand their logs, but interact with them. Supporting multiple services means that LogSense lets users monitor the status of all of their configured nodes grouped by service. And, again, the users have final judgment when deciding if an anomaly is truly an anomaly. In our dashboard, users can view any log lines our ensemble predicted as anomalous and mark them as truly anomalous or actually normal. If the users would like further context on the log, they can click the "More Info" button that takes them to the Loki aggregation. Since LogSense tracks meta data for each log line, a unique log line is swiftly identified, enabling users to view other contextual logs. LogSense's dashboard is updated in real time due to LogSense's stream processing capabilities, which means that the doctors (programmers) can treat their patients (systems) in real time!

## Further questions

Although LogSense provides a usable platform for a majority of users, there remain questions regarding its implementation:

-   Can the theoretical tandem of RAPID and LogGPT prove effective in real world settings?

-   How can a dynamic threshold be implemented to let RAPID make better decisions as time goes on?

-   How can we guard against false negatives?

# Citations

[1] No, G., Lee, Y., Kang, H., and Kang, P., “RAPID: Training-free Retrieval-based Log Anomaly Detection with PLM considering Token-level information”, <i>arXiv e-prints</i>, 2023. doi:10.48550/arXiv.2311.05160.

[2] Han, X., Yuan, S., and Trabelsi, M., “LogGPT: Log Anomaly Detection via GPT”, <i>arXiv e-prints</i>, 2023. doi:10.48550/arXiv.2309.14482.

[3] Landauer, M., Onder, S., Skopik, F., and Wurzenberger, M., “Deep Learning for Anomaly Detection in Log Data: A Survey”, <i>arXiv e-prints</i>, 2022. doi:10.48550/arXiv.2207.03820.

[4] Le, V.-H. and Zhang, H., “Log-based Anomaly Detection with Deep Learning: How Far Are We?”, <i>arXiv e-prints</i>, 2022. doi:10.48550/arXiv.2202.04301.

[5] P. He, J. Zhu, Z. Zheng and M. R. Lyu, "Drain: An Online Log Parsing Approach with Fixed Depth Tree," <i>2017 IEEE International Conference on Web Services (ICWS)<i>, Honolulu, HI, USA, 2017, pp. 33-40, doi: 10.1109/ICWS.2017.13.
