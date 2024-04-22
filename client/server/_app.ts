import { CreateNextContextOptions } from "@trpc/server/adapters/next";
import { serviceRouter } from "./routers/services";
import { logRouter } from "./routers/log";
import { publicProcedure, router } from "./trpc";
import dbConnect from "./db";
import { createKafka } from "./stream";
import { CreateWSSContextFnOptions } from "@trpc/server/adapters/ws";

export const createContext = async (
  opts: CreateNextContextOptions | CreateWSSContextFnOptions
) => {
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
  services: serviceRouter,
});
// Export only the type of a router!
// This prevents us from importing server code on the client.
export type AppRouter = typeof appRouter;
