import { useQueryClient } from "@tanstack/react-query";
import { AnomaliesTable } from "../ui/components/dashboard/AnomaliesTable";
import { RapidLogPrediction } from "../server/routers/log";
import { trpc } from "../utils/trpc";
import { useState } from "react";
import { getQueryKey } from "@trpc/react-query";

export default function Home() {
    const utils = trpc.useUtils();
    const { data: anomalies } = trpc.log.getAll.useQuery(undefined, {
        staleTime: 5 * 1000,
        select: (data) => data?.data,
    });

    trpc.log.onAdd.useSubscription(undefined, {
        onData(log) {
            utils.log.getAll.invalidate();
        },
    });
    return (
        <>
            <h1>Pending Anomalies</h1>
            {anomalies === undefined || anomalies?.length === 0 ? (
                <p>No pending anomalies</p>
            ) : (
                <AnomaliesTable anomalies={anomalies} />
            )}
        </>
    );
}
