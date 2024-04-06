# Logsense - What's to come?

## Software Todo

1. LogGPT integration! Not only creating separate predictions with it, but also ensembling it with RAPID to leverage a mix of token & sequence-level attention. **Ensemble strategy**
2. Make it more "plug-and-play". Keep the installation instructions extremely simple but flexible. 

## Meta Todo

1. Create clean diagrams describing high-level architecture. Consider piecing it out rather than putting it all in a single image.
2. Iterate on the project report. This is an ever-evolving document as long as we keep working on the software and make improvements. 
3. As we wrap up, we need to create a pitch-deck (slides), and a simple demo demonstrating the effectiveness of LogSense on a production system like MapReduce.
4. **Refactor, write tests, and write more documentation!** This is just the standard software engineering lifecycle.

## Ideas to consider

1. To expand upon the "distributed-systems" part, we can train on **traces** to capture the intra-node sequences of logs that describe a particular "event". This is opposed to the single-log model where anomalies can only be predicted on a unique log line. 
	1. HDFS loghub dataset actually consists of traces that are associated with a block ID. 
	2. In my mind, I'm thinking 3 options for structuring logs to display anomaly data:
		1. Non-distributed, unique log line anomalies. This is the default for RAPID.
		2. Non-distributed, log window (1 min chunk) anomalies. This is what LogGPT would predict on if we don't do token-level. 
		3. Distributed, traces across nodes. Each trace contains an ordered sequence of logs. 

## Challenges

1. We need dynamic thresholding. A static threshold would not be acceptable!
2. Coreset, top-k, and other hyperparameters have to be optimized and grid-searched. 
3. False negatives. These never get displayed to the user, so we don't have a way to improve the model by punishing them. **The only solution I can think of is having a more conservative threshold.**
4. Initialization with normal data. Sometimes it just cannot be guarenteed that "training-mode" data will ALL be normal in a very complex system. We need to find better ways to seed the model. 
5. It'd be better if we could add some degree of explainability for our predictions. E.g. for RAPID it could be highlighting the token that shot up the distnace

## Experiments

1. Try out RAPID on single log lines of HDFS and see how well it performs. Does the context really matter? 
2. Try to single out tokens that are farthest from top-k in LogGPT and call those anomalies. Evaluate and see if we can improve how well LogGPT can pinpoint a specific line within a sequence and if we can use that to drive model explainability.