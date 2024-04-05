import Image from "next/image";
import { Inter } from "next/font/google";
import { NavbarSimple } from "../components/NavbarSimple";
import { Button, Code, Table } from "@mantine/core";
import { Layout } from "../components/Layout";
import path from "path";
import { trpc } from "../utils/trpc";
import { useQueryClient } from "@tanstack/react-query";
import { Duration } from "luxon";

interface AnomalousLog {
    hash: string;
    service: string;
    node: string;
    filename: string;
    original_text: string;
    timestamp: number;
    cleaned_text: string;
    score: number;
}

interface AnomaliesTableProps {
    logs: AnomalousLog[];
}

const dateFromUnix = (timestamp: number) => {
    const date = new Date(Number.parseInt(String(timestamp).slice(0, 13)));
    return date.toLocaleString();
};

const AnomalyRow = ({ log }: { log: AnomalousLog }) => {
    const queryClient = useQueryClient();
    const { mutate: markNormal } = trpc.log.markNormal.useMutation({
        onSuccess() {
            queryClient.invalidateQueries(["getNotes"]);
        },
    });
    const { mutate: deleteLog } = trpc.log.delete.useMutation({
        onSuccess() {
            queryClient.invalidateQueries(["getNotes"]);
        },
    });

    const minusTimeRange = (timestamp: number, duration: Duration) => {
        return timestamp - duration.as("milliseconds");
    };

    const plusTimeRange = (timestamp: number, duration: Duration) => {
        return timestamp + duration.as("milliseconds");
    };

    const lokiDataSourceId = "fdhrn6s5cd62oa";
    const panes = {
        tr1: {
            datasource: lokiDataSourceId,
            queries: [
                {
                    refId: "A",
                    expr: `{service="${log.service}", node="${log.node}", filename="${log.filename}"}`,
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
                        Number.parseInt(log.timestamp.toString().slice(0, 13)),
                        Duration.fromObject({ hours: 1 })
                    )
                ),
                to: String(
                    plusTimeRange(
                        Number.parseInt(log.timestamp.toString().slice(0, 13)),
                        Duration.fromObject({ hours: 1 })
                    )
                ),
            },
        },
    };

    console.log(panes.tr1.range);

    const params = new URLSearchParams();
    params.set("orgId", "1");
    params.set("schemaVersion", "1");
    params.set("panes", JSON.stringify(panes));
    const grafanaUrl = "http://localhost:3030/explore?" + params.toString();

    return (
        <Table.Tr key={log.hash}>
            <Table.Td style={{ whiteSpace: "normal" }}>{log.service}</Table.Td>
            <Table.Td>{log.node}</Table.Td>
            <Table.Td>{path.basename(log.filename)}</Table.Td>
            <Table.Td>{dateFromUnix(log.timestamp)}</Table.Td>
            <Table.Td>
                <Code>{log.original_text}</Code>
            </Table.Td>
            <Table.Td>
                <Button
                    variant="light"
                    color="green"
                    onClick={() => markNormal(log.hash)}
                >
                    Mark Normal
                </Button>
            </Table.Td>
            <Table.Td>
                <Button
                    variant="filled"
                    color="red"
                    onClick={() => deleteLog(log.hash)}
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

export const AnomaliesTable = ({ logs }: AnomaliesTableProps) => {
    const rows = (logs || []).map((log) => <AnomalyRow log={log} />);

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
