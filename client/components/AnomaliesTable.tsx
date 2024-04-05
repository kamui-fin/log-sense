import Image from "next/image";
import { Inter } from "next/font/google";
import { NavbarSimple } from "../components/NavbarSimple";
import { Button, Code, Table } from "@mantine/core";
import { Layout } from "../components/Layout";
import path from "path";
import { trpc } from "../utils/trpc";
import { useQueryClient } from "@tanstack/react-query";

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

export const AnomaliesTable = ({ logs }: AnomaliesTableProps) => {
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
    const rows = (logs || []).map((log) => (
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
                <Button variant="light" color="cyan">
                    Inspect
                </Button>
            </Table.Td>
        </Table.Tr>
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
