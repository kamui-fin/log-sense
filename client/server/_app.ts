import { CreateNextContextOptions } from "@trpc/server/adapters/next";
import { configRouter } from "./routers/config";
import { logRouter } from "./routers/log";
import { publicProcedure, router } from "./trpc";
import dbConnect from "./db";
import { createKafka } from "./stream";

export const createContext = async (opts: CreateNextContextOptions) => {
    await dbConnect();
    const kafkaConn = createKafka();
    const producer = kafkaConn.producer();
    await producer.connect();
    return {
        kafkaConn: kafkaConn,
        kafkaProducer: producer,
    };
};

export type Context = Awaited<ReturnType<typeof createContext>>;

export const appRouter = router({
    log: logRouter,
    config: configRouter,
});
// Export only the type of a router!
// This prevents us from importing server code on the client.
export type AppRouter = typeof appRouter;
