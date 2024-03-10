# Log Sense - Research Proposal

In this study, we aim to explore the SoTA (state-of-the-art) LLM-based log anomaly detection models, and empirically evaluate & apply them under a real-time distributed environment.

Specifically, this study can be broken down into 3 parts:
1. Comprehensive evaluation of the models following Le & Zhang's work.
2. Case study for real application within a real-time distributed system.
3. Attempting to answer a few related research questions

We selected 3 models as the focus of our study:

1. LAnoBERT - BERT based [1]
2. LogGPT - GPT based w/ reinforcement learning [2]
3. RAPID - Retrieval-oriented (train-free) [3]

While the benchmarks provided in the original papers are quite convincing of their performance, we wanted to apply them under a less ideal, but highly real-world setting to fully test their capabilities. Overall, we wanted to gain a better understanding of how they could be integrated into existing production systems.

## Applications within distributed systems

With some inspiration from the HDFS [6] dataset, we hypothesize that similar performance could be achieved for similar distributed services by leveraging *session window grouping* [5]. This involves grouping logs based on session identifiers, such as the node or block. 

Following, we will construct a case study by demonstrating the application of the models real-time for a system of microservices and document relevant metrics such as inference times.

### Ensembling

Throughout the course of this study, we wish to explore the effective of additional techniques to enable smooth online (real-time) detection using these models, especially **ensembling**.

An example ensembled prediction process we imagine could work:

1. Initial fast detection with RAPID [3]
2. Update confidence of detection results slowly by running an ensemble of LogGPT [2] and LAnoBERT [1]

This attempts to reach a consensus between real-time speed and accuracy. 

## Empirical evaluation following Le & Zhang

Le & Zhang's paper [5] conducted an intriguing and comprehensive analyis of the older models such as DeepLog and LogAnomaly by investigating their generalizable capabilities under different non-ideal experimental settings. This included:
- Different data selection strategies
- Varying imbalanced class distributions
- Performance under X% of mislabeled logs

The paper's findings reported several pending challenges when adapting to real-world systems:
1. Limited labeled data 
2. Anomalies should be predicted as early as possible
3. Evolving logs following evolving software
4. Log parsing errors severely impact detection performance

With inspiration from the paper, we attempt to conduct a similar investigation on the three mentioned recent models by benchmarking under the experimental settings, incorporating the paper's *specificity* metric as well.

## Research questions

As we conduct the case study and evaluation, we will attempt to try to answer some key questions with empirical data.

### Pre-training applied on out-of-domain logs

A challenge that remains is that these semi-supervised models must only be trained on normal data, thus requiring labels. In this domain, the number of labeled log datasets is rather scarce, so this problem becomes increasingly important to answer. 

How well does pre-training on diverse log data generalize to different data? 

# Citations

[1] Lee, Y., Kim, J., and Kang, P., “LAnoBERT: System Log Anomaly Detection based on BERT Masked Language Model”, <i>arXiv e-prints</i>, 2021. doi:10.48550/arXiv.2111.09564.

[2] Han, X., Yuan, S., and Trabelsi, M., “LogGPT: Log Anomaly Detection via GPT”, <i>arXiv e-prints</i>, 2023. doi:10.48550/arXiv.2309.14482.

[3] No, G., Lee, Y., Kang, H., and Kang, P., “RAPID: Training-free Retrieval-based Log Anomaly Detection with PLM considering Token-level information”, <i>arXiv e-prints</i>, 2023. doi:10.48550/arXiv.2311.05160.

[4] Landauer, M., Onder, S., Skopik, F., and Wurzenberger, M., “Deep Learning for Anomaly Detection in Log Data: A Survey”, <i>arXiv e-prints</i>, 2022. doi:10.48550/arXiv.2207.03820.

[5] Le, V.-H. and Zhang, H., “Log-based Anomaly Detection with Deep Learning: How Far Are We?”, <i>arXiv e-prints</i>, 2022. doi:10.48550/arXiv.2202.04301.

[6] Zhu, J., He, S., He, P., Liu, J., and Lyu, M. R., “Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics”, <i>arXiv e-prints</i>, 2020. doi:10.48550/arXiv.2008.06448.