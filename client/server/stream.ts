import EventEmitter from "events";
import { Kafka } from "kafkajs";
import { LogPrediction } from "./routers/log";
import { LogModel } from "./models/anomalyLog";

const kafka = new Kafka({
    clientId: "logsense-client",
    brokers: ["localhost:9092"],
});
const consumer = kafka.consumer({ groupId: "logsense-client" }); // TODO: research more about group ID
export const eventEmitter = new EventEmitter()

export const initKafkaListener = async () => {
    console.log('Listening to Kafka topic...')
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
            const logPrediction: LogPrediction = JSON.parse(
                message.value.toString()
            );
            if (logPrediction.is_anomaly) {
                console.log(`Anomaly detected: ${logPrediction.cleaned_text}`);
                const {
                    score,
                    is_anomaly,
                    original_text,
                    cleaned_text,
                    hash,
                    timestamp,
                    filename,
                    service,
                    node,
                } = logPrediction;

                // const newLog = new LogModel({
                //     original_text,
                //     cleaned_text,
                //     hash,
                //     timestamp,
                //     filename,
                //     service,
                //     node,
                //     score,
                // });
                // const savedLog = await newLog.save();
                // notify all currently connected subscribers of the update
                eventEmitter.emit("add", logPrediction);
            }
        },
    });
}