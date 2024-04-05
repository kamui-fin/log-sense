import { observable } from "@trpc/server/observable";
import { Log, LogModel } from "../models/anomalyLog";
import { publicProcedure, router } from "../trpc";
import { z } from "zod";
import { eventEmitter } from "../stream";
import { Context } from "../_app";
import { Producer } from "kafkajs";

// TODO: fix schema for service config

export interface LogPrediction {
    score: number;
    is_anomaly: boolean;
    original_text: string;
    cleaned_text: string;
    hash: string; // truncated SHA hash of cleaned_text
    timestamp: number;
    filename: string; // full path
    service: string;
    node: string;
}

const getLogsController = async () => {
    try {
        const logs = await LogModel.find();
        console.log(logs);
        return {
            status: "OK",
            data: logs,
        };
    } catch (error) {
        console.error(error);
    }
};

const deleteController = async (hash: string) => {
    try {
        const result = await LogModel.deleteOne({ hash });
        return {
            status: "OK",
            data: result,
        };
    } catch (error) {
        console.error(error);
    }
};

const markNormalController = async (hash: string, producer: Producer) => {
    try {
        const log = await LogModel.findOne({ hash });
        if (!log) {
            return {
                status: "ERROR",
                message: "Log not found",
            };
        }
        await LogModel.deleteOne({ hash });
        // send to kafka topic for marking normal
        const message = {
            value: JSON.stringify(log),
        };
        producer.send({ topic: "mark-normal", messages: [message] });

        return {
            status: "OK",
        };
    } catch (error) {
        console.error(error);
    }
};

export const logRouter = router({
    getAll: publicProcedure.query(getLogsController),
    markNormal: publicProcedure
        .input(z.string())
        .mutation(async ({ input, ctx }) =>
            markNormalController(input, ctx.kafkaProducer)
        ),
    delete: publicProcedure
        .input(z.string())
        .mutation(async ({ input }) => deleteController(input)),
    onAdd: publicProcedure.subscription(() => {
        console.log("Requesting subscription");
        return observable<LogPrediction>((emit) => {
            const onAdd = (data: LogPrediction) => {
                console.log(data);
                emit.next(data);
            };
            eventEmitter.on("add", onAdd);
            return () => {
                eventEmitter.off("add", onAdd);
            };
        });
    }),
});
