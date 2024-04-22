import { AddService } from "./AddServiceModal";
import {
    ActionIcon,
    Button,
    Modal,
    NumberInput,
    Select,
    Switch, 
    TextInput,
} from "@mantine/core";
import { IconPlus } from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";



export const ListHeader = () => {
    const [opened, { open, close }] = useDisclosure(false);

    return(
        <div>
            <div className="flex gap-8 align-middle items-center">
                <h1>Manage Services</h1>
                <ActionIcon
                    variant="gradient"
                    size="xl"
                    aria-label="Gradient action icon"
                    gradient={{ from: "blue", to: "cyan", deg: 90 }}
                    onClick={open}
                >
                    <IconPlus />
                </ActionIcon>
            </div>
            <AddService opened={opened} open={open} close={close} />
        </div>
    );
}