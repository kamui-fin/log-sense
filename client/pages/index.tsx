import { useQueryClient } from "@tanstack/react-query";
import { AnomaliesTable } from "../ui/components/dashboard/AnomaliesTable";
import { RapidLogPrediction } from "../server/routers/log";
import { trpc } from "../utils/trpc";
import { useState } from "react";

export default function Home() {
  const queryClient = useQueryClient();
  const { data: anomalies } = trpc.log.getAll.useQuery("getLogs", {
    staleTime: 5 * 1000,
    select: (data) => data?.data,
  });

  trpc.log.onAdd.useSubscription(undefined, {
    onData(log) {
      console.log(log);
      queryClient.invalidateQueries(["getLogs"]);
    },
  });
  return (
    <>
      <h1>Pending Anoualies</h1>
      <AnomaliesTable anomalies={anomalies} />
    </>
  );
}
