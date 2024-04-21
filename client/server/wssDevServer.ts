// Standalone websocket server for streaming routes

import ws from "ws";
import { applyWSSHandler } from "@trpc/server/adapters/ws";
import { initKafkaListener } from "./stream";
import { appRouter, createContext } from "./_app";

const WS_PORT = 3002;

const wss = new ws.Server({
    port: WS_PORT,
});

const handler = applyWSSHandler({
    wss,
    router: appRouter,
    createContext,
});

wss.on("connection", (ws) => {
    console.log(`➕➕ Connection (${wss.clients.size})`);
    ws.once("close", () => {
        console.log(`➖➖ Connection (${wss.clients.size})`);
    });
});

console.log(`✅ WebSocket Server listening on ws://localhost:${WS_PORT}`);

initKafkaListener();

process.on("SIGTERM", () => {
    console.log("SIGTERM");
    handler.broadcastReconnectNotification();
    wss.close();
});
