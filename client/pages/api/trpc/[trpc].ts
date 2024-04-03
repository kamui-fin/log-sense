import { createNextApiHandler } from "@trpc/server/adapters/next";
import { appRouter } from "../../../server/_app";

// @link https://nextjs.org/docs/api-routes/introduction
export default createNextApiHandler({
  router: appRouter,
});
