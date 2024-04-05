import {
    CreateNextContextOptions,
    createNextApiHandler,
} from "@trpc/server/adapters/next";
import { appRouter, createContext } from "../../../server/_app";

// @link https://nextjs.org/docs/api-routes/introduction
export default createNextApiHandler({
    router: appRouter,
    createContext,
});
