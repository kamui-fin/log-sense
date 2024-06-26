import Image from "next/image";
import { Inter } from "next/font/google";
import { NavbarSimple } from "../nav/NavbarSimple";
import { Button, Code, Table, stylesToString } from "@mantine/core";
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
    const date = new Date(0);
    date.setUTCSeconds(Number.parseInt(String(timestamp)));
    return date.toLocaleString();
};

const AnomalyRow = ({ log }: { log: UnionAnomalousLog }) => {
    const utils = trpc.useUtils();
    const { mutate: markNormal } = trpc.log.markNormal.useMutation({
        onSuccess() {
            utils.log.getAll.invalidate();
        },
    });
    const { mutate: confirmAnomaly } = trpc.log.confirmAnomaly.useMutation({
        onSuccess() {
            utils.log.getAll.invalidate();
        },
    });

    const minusTimeRange = (timestamp: number, duration: Duration) => {
        return (timestamp - duration.as("seconds")) * 1000;
    };

    const plusTimeRange = (timestamp: number, duration: Duration) => {
        return (timestamp + duration.as("seconds")) * 1000;
    };

    const panes = {
        tr1: {
            datasource: "loki",
            queries: [
                {
                    refId: "A",
                    expr: `{service="${log.service}", node="${log.nodes.join(
                        "|"
                    )}", filename="${log.uniqueFilenames.join("|")}"}`,
                    queryType: "range",
                    datasource: {
                        type: "loki",
                        uid: "loki",
                    },
                    editorMode: "builder",
                    legendFormat: "",
                },
            ],
            range: {
                from: String(
                    minusTimeRange(
                        log.startTimestamp,
                        Duration.fromObject({
                            minutes: Number.parseInt(
                                localStorage.getItem(
                                    "grafanaContextTimeRadius"
                                ) || "1"
                            ),
                        })
                    )
                ),
                to: String(
                    plusTimeRange(
                        log.endTimestamp,
                        Duration.fromObject({
                            minutes: Number.parseInt(
                                localStorage.getItem(
                                    "grafanaContextTimeRadius"
                                ) || "1"
                            ),
                        })
                    )
                ),
            },
        },
    };

    const params = new URLSearchParams();
    params.set("orgId", "1");
    params.set("schemaVersion", "1");
    params.set("panes", JSON.stringify(panes));

    const grafanaUrl =
        `${process.env.NEXT_PUBLIC_GRAFANA_URI}/explore?` + params.toString();

    return (
        <Table.Tr key={log.hash}>
            <Table.Td style={{ whiteSpace: "normal" }}>{log.service}</Table.Td>
            <Table.Td>{log.nodes[0]}</Table.Td>
            <Table.Td>{path.basename(log.uniqueFilenames[0])}</Table.Td>
            <Table.Td>{dateFromUnix(log.endTimestamp)}</Table.Td>
            <Table.Td>
                <Code
                    block={true}
                    style={{
                        maxWidth: "800px",
                    }}
                >
                    {log.text.length > 800
                        ? log.text.slice(0, 800) + "..."
                        : log.text}
                </Code>
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
