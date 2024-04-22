import { ServiceCardProps } from "./ServiceCard";
import { ServiceHeader } from "./ServiceHeader";
import { ListServices } from "./ListServices";
import { useState } from "react";
import { PasswordInput, Tooltip, Center, Text, rem } from "@mantine/core";
import { IconInfoCircle } from "@tabler/icons-react";
import { useForm, zodResolver } from "@mantine/form";
import { useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { trpc } from "../../../utils/trpc";
import { Button, Modal, NumberInput, Switch, TextInput } from "@mantine/core";
import { Tabs } from "@mantine/core";
import {
    IconPhoto,
    IconMessageCircle,
    IconSettings,
} from "@tabler/icons-react";
import { GeneralTab } from "./tabs/ServiceGeneral";
import { RAPIDTab } from "./tabs/ServiceRAPID";
import { GPTTab } from "./tabs/ServiceGPT";
import { useEffect, useRef } from "react";
import { RegexTab } from "./tabs/ServiceRegex";

export const ExtendedView = ({ service, onGoBack }: ServiceCardProps) => {
    const iconStyle = { width: rem(12), height: rem(12) };
    return (
        <div>
            <ServiceHeader name={service.name} goBack={onGoBack} />
            <Tabs defaultValue={"general"}>
                <Tabs.List>
                    <Tabs.Tab value="general">General</Tabs.Tab>
                    <Tabs.Tab value="rapid">RAPID</Tabs.Tab>
                    <Tabs.Tab value="gpt">LogGPT</Tabs.Tab>
                    <Tabs.Tab value="regex">REGEX</Tabs.Tab>
                </Tabs.List>
                <Tabs.Panel value="general">
                    <GeneralTab service={service} onGoBack={onGoBack} />
                </Tabs.Panel>
                <Tabs.Panel value="rapid">
                    <RAPIDTab service={service} onGoBack={onGoBack} />
                </Tabs.Panel>
                <Tabs.Panel value="gpt">
                    <GPTTab service={service} onGoBack={onGoBack} />
                </Tabs.Panel>
                <Tabs.Panel value="regex">
                    <RegexTab service={service} />
                </Tabs.Panel>
            </Tabs>
        </div>
    );
};
