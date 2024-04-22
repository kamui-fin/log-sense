import {
    IconBookmark,
    IconEdit,
    IconHeart,
    IconShare,
    IconTrash,
} from "@tabler/icons-react";
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
    useMantineTheme,
    rem,
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
    onConfigClick?: any;
    onGoBack: any;
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
    const theme = useMantineTheme();

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
        <Card
            shadow="sm"
            padding="lg"
            radius="md"
            withBorder
            className={classes.card}
        >
            {/* <Overlay className={classes.overlay} zIndex={0} /> */}

            <div className={classes.content}>
                <Text size="lg" fw={700} className={classes.title}>
                    {name}
                </Text>

                <Text size="sm" className={classes.description}>
                    {description}
                </Text>

                <Group gap={8} mr={0} className={classes.btnGroup}>
                    <ActionIcon
                        radius="md"
                        size={36}
                        className={classes.action}
                        onClick={() => handleConfigClick(service)}
                    >
                        <IconEdit
                            style={{ width: rem(16), height: rem(16) }}
                            color={theme.colors.blue[6]}
                        />
                    </ActionIcon>
                    <ActionIcon
                        className={classes.action}
                        radius="md"
                        size={36}
                        onClick={() => onDeleteHandler(service._id)}
                    >
                        <IconTrash
                            style={{ width: rem(16), height: rem(16) }}
                            color={theme.colors.red[6]}
                        />
                    </ActionIcon>
                </Group>
            </div>
        </Card>
    );
}
