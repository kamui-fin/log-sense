import { observable } from "@trpc/server/observable";
import { Log, LogModel } from "../models/anomalyLog";
import { publicProcedure, router } from "../trpc";
import { z } from "zod";
import { eventEmitter } from "../stream";

// TODO: fix schema for service config

export interface LogPrediction {
    score: number;
    is_anomaly: boolean;
    original_text: string;
    cleaned_text: string;
    hash: number; // truncated SHA hash of cleaned_text
    timestamp: number;
    filename: string; // full path
    service: string;
    node: string;
}

export const logRouter = router({
    onAdd: publicProcedure.subscription(() => {
        console.log('Requesting subscription')
        return observable<LogPrediction>((emit) => {
            const onAdd = (data: LogPrediction) => {
                console.log(data)
                emit.next(data);
            };
            eventEmitter.on("add", onAdd);
            return () => {
                eventEmitter.off("add", onAdd);
            };
        });
    }),
});
