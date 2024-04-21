import EventEmitter from "events";
import { Kafka } from "kafkajs";
import {
    RapidLogPrediction,
    GptLogPrediction,
    rapidToUnified,
    logGptToUnified,
} from "./routers/log";
import { RapidLogModel } from "./models/rapid_log";
import dbConnect from "./db";
import { GptLogModel } from "./models/gpt_log";

export const eventEmitter = new EventEmitter();

const KAFKA_BROKER_HOST = process.env.KAFKA_BROKER_URI || "localhost:9092";

const CLIENT_ID = "logsense-client";
const GROUP_ID = "logsense";

export const createKafka = () => {
    const kafka = new Kafka({
        clientId: CLIENT_ID,
        brokers: [KAFKA_BROKER_HOST],
    });
    return kafka;
};

const kafka = createKafka();
const consumer = kafka.consumer({ groupId: GROUP_ID });

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
        eachMessage: async ({
            topic,
            partition,
            message,
            heartbeat,
            pause,
        }) => {
            if (message.value === null) return;
            const logPrediction: BaseLogPrediction = JSON.parse(
                message.value.toString()
            );
            // handle rapid specific logic here
            let unifiedNewLog;
            if (logPrediction.is_anomaly && logPrediction.type === "rapid") {
                const prediction: RapidLogPrediction = JSON.parse(
                    message.value.toString()
                );
                console.log(`Anomaly detected: ${prediction.cleaned_text}`);
                const newLog = new RapidLogModel({ ...prediction });
                const savedLog = await newLog.save();
                console.log(savedLog);
                unifiedNewLog = rapidToUnified(savedLog);
            } else if (logPrediction.type === "log_gpt") {
                const newLog = new GptLogModel(
                    JSON.parse(message.value.toString())
                );
                const savedLog = await newLog.save();
                unifiedNewLog = logGptToUnified({
                    ...savedLog,
                    _id: savedLog._id.toString(),
                    type: "log_gpt",
                });
            } else {
                console.error("Invalid log prediction type");
                return;
            }
            // notify all currently connected subscribers of the update
            eventEmitter.emit("add", unifiedNewLog);
        },
    });
};
