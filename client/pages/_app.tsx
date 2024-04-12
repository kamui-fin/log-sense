import "@mantine/core/styles.css";
import "@/components/styles/globals.css";
import type { AppProps } from "next/app";
import { withTRPC } from "@trpc/next";
import { createTheme, MantineProvider } from "@mantine/core";
import { Layout } from "../ui/Layout";
import { AppRouter } from "../server/_app";
import { trpc } from "../utils/trpc";

const theme = createTheme({
    /** Put your mantine theme override here */
});

function App({ Component, pageProps }: AppProps) {
    return (
        <MantineProvider theme={theme}>
            <Layout>
                <Component {...pageProps} />
            </Layout>
        </MantineProvider>
    );
}

export default trpc.withTRPC(App);
