import { TRPCLink, httpBatchLink, loggerLink, splitLink } from "@trpc/client";
import { createTRPCNext } from "@trpc/next";
import type { AppRouter } from "../server/_app";
import { createTRPCClient, createWSClient, wsLink } from "@trpc/client";
import { NextPageContext } from "next";
import superjson from "superjson";

function getBaseUrl() {
    if (typeof window !== "undefined")
        // browser should use relative path
        return "";

    if (process.env.VERCEL_URL)
        // reference for vercel.com
        return `https://${process.env.VERCEL_URL}`;

    if (process.env.RENDER_INTERNAL_HOSTNAME)
        // reference for render.com
        return `http://${process.env.RENDER_INTERNAL_HOSTNAME}:${process.env.PORT}`;

    // assume localhost
    return `http://localhost:${process.env.PORT ?? 3000}`;
}

function getEndingLink(ctx: NextPageContext | undefined): TRPCLink<AppRouter> {
    const batchLink = httpBatchLink({
        /**
         * @link https://trpc.io/docs/v11/data-transformers
         */
        transformer: superjson,
        url: `${getBaseUrl()}/api/trpc`,
        headers() {
            if (!ctx?.req?.headers) {
                return {};
            }
            // on ssr, forward client's headers to the server
            return {
                ...ctx.req.headers,
                "x-ssr": "1",
            };
        },
    });
    const websocketLink = wsLink({
        client: createWSClient({ url: `ws://localhost:3001` }),
        /**
         * @link https://trpc.io/docs/v11/data-transformers
         */
        transformer: superjson,
    });
    return splitLink({
        condition: (op) => op.type === "subscription",
        true: websocketLink,
        false: batchLink,
    });
}

/**
 * A set of strongly-typed React hooks from your `AppRouter` type signature with `createReactQueryHooks`.
 * @link https://trpc.io/docs/v11/react#3-create-trpc-hooks
 */
export const trpc = createTRPCNext<AppRouter>({
    /**
     * @link https://trpc.io/docs/v11/ssr
     */
    ssr: false,
    // ssrPrepass,
    config({ ctx }) {
        /**
         * If you want to use SSR, you need to use the server's full URL
         * @link https://trpc.io/docs/v11/ssr
         */

        return {
            /**
             * @link https://trpc.io/docs/v11/client/links
             */
            links: [
                // adds pretty logs to your console in development and logs errors in production
                loggerLink({
                    enabled: (opts) =>
                        (process.env.NODE_ENV === "development" &&
                            typeof window !== "undefined") ||
                        (opts.direction === "down" &&
                            opts.result instanceof Error),
                }),
                getEndingLink(ctx),
            ],
            /**
             * @link https://tanstack.com/query/v5/docs/reference/QueryClient
             */
            queryClientConfig: {
                defaultOptions: { queries: { staleTime: 60 } },
            },
        };
    },
    /**
     * @link https://trpc.io/docs/v11/data-transformers
     */
    transformer: superjson,
});
