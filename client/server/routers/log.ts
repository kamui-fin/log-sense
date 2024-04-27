import { observable } from "@trpc/server/observable";
import { RapidLog, RapidLogModel } from "../models/rapid_log";
import { publicProcedure, router } from "../trpc";
import { z } from "zod";
import { eventEmitter } from "../stream";
import { Context } from "../_app";
import { Producer } from "kafkajs";
import { GptLogModel, LogGptChunk } from "../models/gpt_log";
import { ObjectId } from "mongoose";
import * as mongoose from "mongoose";
import { log } from "console";

export interface GptLogPrediction {
    _id: string;
    hash: string;
    type: "log_gpt";
    is_anomaly: boolean;
    chunk: LogGptChunk;
    service: string; // copy of the service in the first log
    original_logs: RapidLog[];
}

export interface RapidLogPrediction extends RapidLog {
    type: "rapid";
}

// we need an intermediary type to handle the union of rapid and gpt logs
export interface UnionAnomalousLog {
    type: "rapid" | "log_gpt";
    id: string; // point to rapid _id or the window id for log_gpt
    // global hash in case of gpt over the entire window text
    hash: string;
    service: string; // a window log
    nodes: string[]; // wait it could be multiple nodes if we're using traces
    uniqueFilenames: string[]; // we could have multiple files in a window
    startTimestamp: number;
    endTimestamp: number;
    text: string; // for window, we just \n join
}

export const rapidToUnified = (rapidLog: RapidLog): UnionAnomalousLog => {
    const service = rapidLog.service;
    const nodes = [rapidLog.node];
    const uniqueFilenames = [rapidLog.filename];
    const startTimestamp = rapidLog.timestamp;
    const endTimestamp = rapidLog.timestamp;

    const id = rapidLog._id.toString();
    const hash = rapidLog.hash;
    const text = rapidLog.original_text;

    const log: UnionAnomalousLog = {
        id,
        type: "rapid",
        hash,
        service,
        nodes,
        uniqueFilenames,
        startTimestamp,
        endTimestamp,
        text,
    };

    return log;
};

export const logGptToUnified = (window: GptLogPrediction) => {
    const text = window.original_logs
        .map((log) => log.original_text)
        .join("\n");
    const service = window.service;
    const nodes = Array.from(
        new Set(window.original_logs.map((log) => log.node))
    );
    const uniqueFilenames = Array.from(
        new Set(window.original_logs.map((log) => log.filename))
    );
    const startTimestamp = window.original_logs[0].timestamp;
    const endTimestamp =
        window.original_logs[window.original_logs.length - 1].timestamp;

    const id = window._id.toString();
    const hash = window.hash;

    const log: UnionAnomalousLog = {
        id,
        type: "log_gpt",
        hash,
        service,
        nodes,
        uniqueFilenames,
        startTimestamp,
        endTimestamp,
        text,
    };

    return log;
};

const getLogsController = async () => {
    try {
        // Query rapid and log_gpt collections s.t is_anomaly = true and sort by decreasing timestamp
        const rapidLogs = await RapidLogModel.find({
            is_anomaly: true,
            prompt_user: true,
        }).sort({ timestamp: -1 });

        const gptLogs: GptLogPrediction[] = (
            await GptLogModel.find(
                {
                    is_anomaly: true,
                    prompt_user: true,
                },
                { _id: 1, hash: 1, service: 1, original_logs: 1 }
            ).sort({ timestamp: -1 })
        ).map((log) => ({ ...log.toObject(), type: "log_gpt" }));

        const rapidOriginalLogs = rapidLogs.map(rapidToUnified);
        const gptOriginalLogs = gptLogs.flatMap(logGptToUnified);
        const anomalies = [...rapidOriginalLogs, ...gptOriginalLogs].sort(
            (a, b) => b.endTimestamp - a.endTimestamp
        );
        return {
            status: "OK",
            data: anomalies,
        };
    } catch (error) {
        console.error(error);
    }
};

const markNormalController = async (
    _id: string,
    type: "rapid" | "log_gpt",
    producer: Producer
) => {
    try {
        let updateResult;
        if (type === "log_gpt") {
            // For LogGPT: we must also mark it for finetuning
            updateResult = await GptLogModel.updateOne(
                { _id },
                { is_anomaly: false, train_strategy: "finetune" }
            );
        } else {
            updateResult = await RapidLogModel.updateOne(
                { _id },
                { is_anomaly: false }
            );

            // Only for RAPID: send to mark-normal topic for instant training
            const log = await RapidLogModel.findOne({ _id });
            const message = {
                value: JSON.stringify(log),
            };
            producer.send({ topic: "mark-normal", messages: [message] });
        }

        if (updateResult.modifiedCount === 0) {
            return {
                status: "ERROR",
                message: "No document found",
            };
        }
        return {
            status: "OK",
            data: updateResult,
        };
    } catch (error) {
        console.error(error);
    }
};

// confirm anomaly controller simply marks the prompt_user flag to false depending on the type of model
const confirmAnomalyController = async (
    _id: string,
    type: "rapid" | "log_gpt"
) => {
    try {
        let updateResult;
        if (type === "log_gpt") {
            updateResult = await GptLogModel.updateOne(
                { _id },
                { prompt_user: false }
            );
        } else {
            updateResult = await RapidLogModel.updateOne(
                { _id },
                { prompt_user: false }
            );
        }

        if (updateResult.modifiedCount === 0) {
            return {
                status: "ERROR",
                message: "No document found",
            };
        }
        return {
            status: "OK",
            data: updateResult,
        };
    } catch (error) {
        console.error(error);
    }
};

export const logRouter = router({
    getAll: publicProcedure.query(getLogsController),
    confirmAnomaly: publicProcedure
        .input(
            z.object({
                _id: z.string(),
                type: z.union([z.literal("rapid"), z.literal("log_gpt")]),
            })
        )
        .mutation(async ({ input }) =>
            confirmAnomalyController(input._id, input.type)
        ),
    markNormal: publicProcedure
        .input(
            z.object({
                _id: z.string(),
                type: z.union([z.literal("rapid"), z.literal("log_gpt")]),
            })
        )
        .mutation(async ({ input, ctx }) =>
            markNormalController(input._id, input.type, ctx.kafkaProducer)
        ),
    onAdd: publicProcedure.subscription(() => {
        return observable<UnionAnomalousLog>((emit) => {
            const onAdd = (data: UnionAnomalousLog) => {
                emit.next(data);
            };
            eventEmitter.on("add", onAdd);
            return () => {
                eventEmitter.off("add", onAdd);
            };
        });
    }),
});
