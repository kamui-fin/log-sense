import { useQueryClient } from "@tanstack/react-query";
import { AnomaliesTable } from "../components/AnomaliesTable";
import { LogPrediction } from "../server/routers/log";
import { trpc } from "../utils/trpc";
import { useState } from "react";

export default function Home() {
    const queryClient = useQueryClient();
    const { data: logs } = trpc.log.getAll.useQuery("getLogs", {
        staleTime: 5 * 1000,
        select: (data) => data?.data,
    });
    console.log(logs);

    trpc.log.onAdd.useSubscription(undefined, {
        onData(log) {
            queryClient.invalidateQueries(["getLogs"]);
        },
    });
    return (
        <>
            <h1>Pending Anomalies</h1>
            <AnomaliesTable logs={logs} />
        </>
    );
}
