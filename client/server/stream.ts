import EventEmitter from "events";
import { Kafka } from "kafkajs";
import { RapidLogPrediction, GptLogPrediction } from "./routers/log";
import { RapidLogModel } from "./models/rapid_log";
import dbConnect from "./db";
import { GptLogModel } from "./models/gpt_log";

export const eventEmitter = new EventEmitter();

export const createKafka = () => {
  const kafka = new Kafka({
    clientId: "logsense-client",
    brokers: ["localhost:9092"],
  });
  return kafka;
};

const kafka = createKafka();
const consumer = kafka.consumer({ groupId: "logsense-client" }); // TODO: research more about group ID

// typescript type where we only know that "type": "rapid" | "gpt"
interface BaseLogPrediction {
  type: "rapid" | "log_gpt";
  is_anomaly: boolean;
}

export const initKafkaListener = async () => {
  await dbConnect();
  console.log("Listening to Kafka topic...");
  await consumer.connect();
  await consumer.subscribe({ topic: "predictions" });
  await consumer.run({
    eachMessage: async ({ topic, partition, message, heartbeat, pause }) => {
      if (message.value === null) return;
      const logPrediction: BaseLogPrediction = JSON.parse(
        message.value.toString()
      );
      // handle rapid specific logic here
      let newLog;
      let unifiedNewLog;
      if (logPrediction.is_anomaly && logPrediction.type === "rapid") {
        const prediction: RapidLogPrediction = JSON.parse(
          message.value.toString()
        );
        console.log(`Anomaly detected: ${prediction.cleaned_text}`);
        newLog = new RapidLogModel({ ...prediction });
      } else if (logPrediction.type === "log_gpt") {
        console.log(JSON.parse(message.value.toString()));
        newLog = new GptLogModel(JSON.parse(message.value.toString()));
        console.log(newLog);
      } else {
        console.error("Invalid log prediction type");
        return;
      }
      const savedLog = await newLog.save();

      // notify all currently connected subscribers of the update
      eventEmitter.emit("add", savedLog);
    },
  });
};
