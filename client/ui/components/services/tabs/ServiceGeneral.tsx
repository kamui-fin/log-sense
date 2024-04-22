import { TextInput, NumberInput, Switch, Textarea } from "@mantine/core";
import { ListServices } from "../ListServices";
import { FormEventHandler, useState } from "react";
import { PasswordInput, Tooltip, Center, Text, rem } from "@mantine/core";
import { IconInfoCircle } from "@tabler/icons-react";
import { useForm, zodResolver } from "@mantine/form";
import { z } from "zod";
import { trpc } from "../../../../utils/trpc";
import { Button, Modal } from "@mantine/core";
import { Tabs } from "@mantine/core";
import {
    IconPhoto,
    IconMessageCircle,
    IconSettings,
} from "@tabler/icons-react";
import { ServiceCardProps } from "../ServiceCard";

type UpdateGeneralInput = z.TypeOf<typeof updateGeneralSchema>;

const updateGeneralSchema = z.object({
    name: z.string(),
    description: z.string(),
    is_train: z.boolean(),
});

export const GeneralTab = ({ service, onGoBack }: ServiceCardProps) => {
    const { name, description, is_train } = service;

    const form = useForm<UpdateGeneralInput>({
        initialValues: {
            name,
            description,
            is_train,
        },
        validate: zodResolver(updateGeneralSchema),
    });

    const utils = trpc.useUtils();
    const { mutate: updateService } = trpc.services.updateService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });

    const submitGeneral = (values: UpdateGeneralInput) => {
        updateService({
            params: { id: service._id },
            body: values,
        });
    };

    return (
        <div>
            <div className="grid grid-cols-3 gap-6 mb-4">
                <form
                    onSubmit={form.onSubmit((values) => submitGeneral(values))}
                >
                    <TextInput
                        label="Name"
                        mt="md"
                        fz="lg"
                        fw={500}
                        {...form.getInputProps("name")}
                        placeholder={name}
                    />
                    <TextInput
                        label="Description"
                        fz="sm"
                        mt="xs"
                        {...form.getInputProps("description")}
                        placeholder={description}
                    />
                    <Switch
                        color="green"
                        label="Training mode"
                        description="All logs are assumed to be normal"
                        size="md"
                        mt="lg"
                        mb="xs"
                        defaultChecked={is_train}
                        {...form.getInputProps("is_train")}
                    />
                    <div className="flex justify-start pt-4">
                        <Button type="submit" radius="md">
                            Update Config
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    );
};
