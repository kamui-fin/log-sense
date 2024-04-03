import Image from "next/image";
import { Inter } from "next/font/google";
import { NavbarSimple } from "../components/NavbarSimple";
import { Button, Code, Table } from "@mantine/core";
import { Layout } from "../components/Layout";
import { trpc } from "../utils/trpc";
import { useState } from "react";

const inter = Inter({ subsets: ["latin"] });

const anomalies = [
  {
    hash: 0,
    service: "HDFS",
    node: "us-west-1",
    filename: "http.log",
    original_text: "2024-03-27 17:46:15.078 [error] electron #2: https://mobile.events.data.microsoft.com/OneCollector/1.0?cors=true&content-type=application/x-json-stream - error POST net::ERR_NETWORK_CHANGED",
    timestamp: 1712112981,

    cleaned_text: "",
    is_anomaly: true,
    score: 0,
  },
  {
    hash: 1,
    service: "Linux",
    node: "us-west-2",
    filename: "main.log",
    text: "Jun 15 12:12:34 combo sshd(pam_unix)[23397]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4",
    timestamp: 1713134981,

    cleaned_text: "",
    is_anomaly: true,
    score: 0,
  },
  {
    hash: 2,
    service: "Linux",
    node: "us-west-2",
    filename: "main.log",
    text: "Jun 15 12:12:34 combo sshd(pam_unix)[23397]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4",
    timestamp: 1713534981,

    cleaned_text: "",
    is_anomaly: true,
    score: 0,
  },
];

const dateFromUnix = (timestamp: number) => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleString();
}

const AnomaliesTable = () => {
  const rows = anomalies.map((log) => (
    <Table.Tr key={log.hash}>
      <Table.Td style={{ whiteSpace: "normal" }}>{log.service}</Table.Td>
      <Table.Td>{log.node}</Table.Td>
      <Table.Td>{log.filename}</Table.Td>
      <Table.Td>{dateFromUnix(log.timestamp)}</Table.Td>
      <Table.Td>
        <Code>{log.text}</Code>
      </Table.Td>
      <Table.Td>
        <Button variant="light" color="green">
          Mark Normal
        </Button>
      </Table.Td>
      <Table.Td>
        <Button variant="filled" color="red">
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
    <Table verticalSpacing="md">
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

export default function Home() {
  const [logs, setLogs] = useState(anomalies)

  trpc.log.onAdd.useSubscription(undefined, {
    onData(log) {
      setLogs([...logs, log])
    }
  })
  return (
    <div>
      <h1>Pending Anomalies</h1>
      <AnomaliesTable />
    </div>
  );
}
