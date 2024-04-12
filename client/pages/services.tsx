import { z } from "zod";
import {
    ActionIcon,
    Button,
    Modal,
    NumberInput,
    Select,
    Switch,
    TextInput,
} from "@mantine/core";
import { Layout } from "../ui/Layout";
import { ListServices } from "../ui/components/services/ListServices";
import { AddService } from "../ui/components/services/AddServiceModal";
import { NavbarSimple } from "../ui/components/nav/NavbarSimple";
import { ServiceCard } from "../ui/components/services/ServiceCard";
import { IconPlus } from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";
import { useForm, zodResolver } from "@mantine/form";
import { useQueryClient } from "@tanstack/react-query";
import { trpc } from "../utils/trpc";

const ManageServices = () => {
    const [opened, { open, close }] = useDisclosure(false);
    return (
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
            <ListServices />
        </div>
    );
};

export default ManageServices;
