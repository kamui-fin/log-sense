import ws from 'ws';
import { applyWSSHandler } from '@trpc/server/adapters/ws';
import { eventEmitter, initKafkaListener } from "./stream";
import { appRouter } from './_app';

const wss = new ws.Server({
    port: 3001,
});

const handler = applyWSSHandler({ wss, router: appRouter });

wss.on('connection', (ws) => {
  console.log(`➕➕ Connection (${wss.clients.size})`);
  ws.once('close', () => {
    console.log(`➖➖ Connection (${wss.clients.size})`);
  });
});

console.log('✅ WebSocket Server listening on ws://localhost:3001');

initKafkaListener();

process.on('SIGTERM', () => {
  console.log('SIGTERM');
  handler.broadcastReconnectNotification();
  wss.close();
});


eventEmitter.on('add', () => {
    console.log("Lmao")
})