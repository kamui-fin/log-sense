import { IconHeart, IconTrash } from "@tabler/icons-react";
import { useForm, zodResolver } from "@mantine/form";
import Link from "next/link";
import {
    Card,
    Image,
    Text,
    Group,
    Badge,
    Button,
    ActionIcon,
    Select,
    NumberInput,
    Overlay,
} from "@mantine/core";
import classes from "./ServiceCard.module.css";
import { Switch } from "@mantine/core";
import { useQueryClient } from "@tanstack/react-query";
import { trpc } from "../../../utils/trpc";
import { z } from "zod";
import { Update } from "next/dist/build/swc";

interface Service {
    _id: string;
    name: string;
    description: string;
    is_train: boolean;
    threshold: number;
    coreset_size: number;
    enable_trace: boolean;
    trace_regex: string;
    top_k: number;
    max_pretrain: number;
    context_size: number;
    lr_pretraining: number;
    lr_finetuning: number;
    train_batch_size: number;
    num_episodes: number;
    num_epochs: number;
    vocab_size: number;
}

export interface ServiceCardProps {
    service: Service;
    onConfigClick: any;
    onSubmitTab: any;
}

const updateServiceSchema = z.object({
    is_train: z.boolean().default(false),
    threshold: z.number().optional(),
    coreset_size: z.number().optional(),
});

type UpdateServiceInput = z.TypeOf<typeof updateServiceSchema>;

export function ServiceCard({ service, onConfigClick }: ServiceCardProps) {
    const { _id, name, description, is_train, threshold, coreset_size } =
        service;
    const handleConfigClick = onConfigClick;

    const form = useForm<UpdateServiceInput>({
        initialValues: {
            is_train,
            threshold,
            coreset_size,
        },
        validate: zodResolver(updateServiceSchema),
    });

    const utils = trpc.useUtils();
    const { mutate: updateService } = trpc.services.updateService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });
    const { mutate: deleteService } = trpc.services.deleteService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });

    const onDeleteHandler = (serviceId: string) => {
        deleteService({ id: serviceId });
    };

    const onUpdateHandler = form.onSubmit((values) => {
        updateService({ params: { id: service._id }, body: values });
    });

    return (
        <Card radius="md" className={classes.card}>
            <Overlay className={classes.overlay} opacity={0.55} zIndex={0} />

            <div className={classes.content}>
                <Text size="lg" fw={700} className={classes.title}>
                    {name}
                </Text>

                <Text size="sm" className={classes.description}>
                    {description}
                </Text>
                <Group mt="xs">
                    <Button
                        onClick={() => handleConfigClick(service)}
                        className={classes.action}
                        variant="white"
                        color="dark"
                        size="xs"
                    >
                        Config
                    </Button>
                    <ActionIcon
                        variant="default"
                        radius="md"
                        size={36}
                        onClick={() => onDeleteHandler(service._id)}
                    >
                        <IconTrash className={classes.like} stroke={1.5} />
                    </ActionIcon>
                </Group>
            </div>
        </Card>
    );
}
