import { configRouter } from "./routers/config";
import { logRouter } from "./routers/log";
import { publicProcedure, router } from "./trpc";

export const appRouter = router({
  log: logRouter,
  config: configRouter
});
// Export only the type of a router!
// This prevents us from importing server code on the client.
export type AppRouter = typeof appRouter;