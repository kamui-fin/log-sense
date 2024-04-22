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

const ManageServicesNew = () => {
    const [opened, { open, close }] = useDisclosure(false);
    return (
        <div>
            <ListServices />
        </div>
    );
};

export default ManageServicesNew;
