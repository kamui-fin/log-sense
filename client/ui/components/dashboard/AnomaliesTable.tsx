import Image from "next/image";
import { Inter } from "next/font/google";
import { NavbarSimple } from "../nav/NavbarSimple";
import { Button, Code, Table } from "@mantine/core";
import { Layout } from "../../Layout";
import path from "path";
import { trpc } from "../../../utils/trpc";
import { useQueryClient } from "@tanstack/react-query";
import { Duration } from "luxon";
import { RapidLog } from "../../../server/models/rapid_log";
import { GptLog } from "../../../server/models/gpt_log";
import { UnionAnomalousLog } from "@/components/server/routers/log";

interface AnomaliesTableProps {
    anomalies: UnionAnomalousLog[];
}

const dateFromUnix = (timestamp: number) => {
    const date = new Date(Number.parseInt(String(timestamp).slice(0, 13)));
    return date.toLocaleString();
};

const AnomalyRow = ({ log }: { log: UnionAnomalousLog }) => {
    const queryClient = useQueryClient();
    const { mutate: markNormal } = trpc.log.markNormal.useMutation({
        onSuccess() {
            queryClient.invalidateQueries({ queryKey: ["getLogs"] });
        },
    });
    const { mutate: confirmAnomaly } = trpc.log.confirmAnomaly.useMutation({
        onSuccess() {
            queryClient.invalidateQueries({ queryKey: ["getLogs"] });
        },
    });

    const minusTimeRange = (timestamp: number, duration: Duration) => {
        return timestamp - duration.as("milliseconds");
    };

    const plusTimeRange = (timestamp: number, duration: Duration) => {
        return timestamp + duration.as("milliseconds");
    };

    const lokiDataSourceId = "adioh7pfmw7i8b"; // TODO: Make this configurable
    const panes = {
        tr1: {
            datasource: lokiDataSourceId,
            queries: [
                {
                    refId: "A",
                    expr: `{service="${log.service}", node="${log.nodes.join(
                        "|"
                    )}", filename="${log.uniqueFilenames.join("|")}"}`,
                    queryType: "range",
                    datasource: {
                        type: "loki",
                        uid: lokiDataSourceId,
                    },
                    editorMode: "builder",
                    legendFormat: "",
                },
            ],
            // TODO: Make context configurable
            range: {
                from: String(
                    minusTimeRange(
                        Number.parseInt(
                            log.startTimestamp.toString().slice(0, 13)
                        ),
                        Duration.fromObject({ hours: 1 })
                    )
                ),
                to: String(
                    plusTimeRange(
                        Number.parseInt(
                            log.endTimestamp.toString().slice(0, 13)
                        ),
                        Duration.fromObject({ hours: 1 })
                    )
                ),
            },
        },
    };

    console.log(panes.tr1.queries[0].expr);

    const params = new URLSearchParams();
    params.set("orgId", "1");
    params.set("schemaVersion", "1");
    params.set("panes", JSON.stringify(panes));
    const grafanaUrl = "http://localhost:3030/explore?" + params.toString();

    return (
        <Table.Tr key={log.hash}>
            <Table.Td style={{ whiteSpace: "normal" }}>{log.service}</Table.Td>
            <Table.Td>{log.nodes[0]}</Table.Td>
            <Table.Td>{path.basename(log.uniqueFilenames[0])}</Table.Td>
            <Table.Td>{dateFromUnix(log.endTimestamp)}</Table.Td>
            <Table.Td>
                <Code>{log.text}</Code>
            </Table.Td>
            <Table.Td>
                <Button
                    variant="light"
                    color="green"
                    onClick={() =>
                        markNormal({
                            type: log.type,
                            _id: log.id,
                        })
                    }
                >
                    Mark Normal
                </Button>
            </Table.Td>
            <Table.Td>
                <Button
                    variant="filled"
                    color="red"
                    onClick={() =>
                        confirmAnomaly({
                            _id: log.id,
                            type: log.type,
                        })
                    }
                >
                    Confirm
                </Button>
            </Table.Td>
            <Table.Td>
                <a href={grafanaUrl}>
                    <Button variant="light" color="cyan">
                        Inspect
                    </Button>
                </a>
            </Table.Td>
        </Table.Tr>
    );
};
export const AnomaliesTable = ({ anomalies }: AnomaliesTableProps) => {
    const rows = (anomalies || []).map((log, index) => (
        <AnomalyRow key={index} log={log} />
    ));

    return (
        <Table verticalSpacing="md" className="w-full">
            <Table.Thead>
                <Table.Tr>
                    <Table.Th>Service</Table.Th>
                    <Table.Th>Node</Table.Th>
                    <Table.Th>Filename</Table.Th>
                    <Table.Th>Timestamp</Table.Th>
                    <Table.Th>Log</Table.Th>
                    <Table.Th>Misclassified</Table.Th>
                    <Table.Th>Clear</Table.Th>
                    <Table.Th>Grafana</Table.Th>
                </Table.Tr>
            </Table.Thead>
            <Table.Tbody>{rows}</Table.Tbody>
        </Table>
    );
};
