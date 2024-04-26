- Train with HDFS tracing setup
- Train on BGL
- Inference tests on both models. Make sure it lines up with expected F1 scores
- Check if UI-side grafana contextualization still works
- With pre-initialized models, test from promtail sources. Try to get a convincing demo out of this

- Test for fault-tolerance by killing kafka, mongo, minio etc