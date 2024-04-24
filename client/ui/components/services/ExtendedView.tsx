import { ServiceCardProps } from "./ServiceCard";
import { ServiceHeader } from "./ServiceHeader";
import { useState } from "react";
import { rem } from "@mantine/core";
import { Tabs } from "@mantine/core";
import { GeneralTab } from "./tabs/ServiceGeneral";
import { RAPIDTab } from "./tabs/ServiceRAPID";
import { GPTTab } from "./tabs/ServiceGPT";
import { RegexTab } from "./tabs/ServiceRegex";

export const ExtendedView = ({ service, onGoBack }: ServiceCardProps) => {
    const [name, setName] = useState(service.name)

    const changeName = (newName: string) => {
        setName(newName)
    }

    const iconStyle = { width: rem(12), height: rem(12) };
    return (
        <div>
            <ServiceHeader name={name} goBack={onGoBack} />
            <Tabs defaultValue={"general"}>
                <Tabs.List>
                    <Tabs.Tab value="general">General</Tabs.Tab>
                    <Tabs.Tab value="rapid">RAPID</Tabs.Tab>
                    <Tabs.Tab value="gpt">LogGPT</Tabs.Tab>
                    <Tabs.Tab value="regex">REGEX</Tabs.Tab>
                </Tabs.List>
                <Tabs.Panel value="general">
                    <GeneralTab service={service} onChangeName = {changeName}/>
                </Tabs.Panel>
                <Tabs.Panel value="rapid">
                    <RAPIDTab service={service}/>
                </Tabs.Panel>
                <Tabs.Panel value="gpt">
                    <GPTTab service={service} />
                </Tabs.Panel>
                <Tabs.Panel value="regex">
                    <RegexTab service={service} />
                </Tabs.Panel>
            </Tabs>
        </div>
    );
};
