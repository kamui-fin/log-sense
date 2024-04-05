import { useForm, zodResolver } from "@mantine/form";
import { useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { trpc } from "../utils/trpc";
import { Button, Modal, NumberInput, Switch, TextInput } from "@mantine/core";

const createServiceSchema = z.object({
    name: z.string(),
    description: z.string(),
    isTrain: z.boolean().default(false),
    threshold: z.number().optional(),
    coresetSize: z.number().optional(),
});

type CreateServiceInput = z.TypeOf<typeof createServiceSchema>;

export const AddService = ({ opened, open, close }) => {
    const queryClient = useQueryClient();
    const { isLoading, mutate: createService } =
        trpc.config.createService.useMutation({
            onSuccess() {
                queryClient.invalidateQueries(["getServices"]);
                close();
            },
        });

    const methods = useForm<CreateServiceInput>({
        resolver: zodResolver(createServiceSchema),
    });

    return (
        <Modal opened={opened} onClose={close} title="Add service">
            <form
                onSubmit={methods.onSubmit((values) => createService(values))}
            >
                <TextInput
                    label="Name"
                    fz="lg"
                    fw={500}
                    {...methods.getInputProps("name")}
                />
                <TextInput
                    label="Description"
                    fz="sm"
                    mt="xs"
                    {...methods.getInputProps("description")}
                />
                <NumberInput
                    label="Coreset Size"
                    description="Number of neighbors to use for RAPID"
                    placeholder="2"
                    mt="xs"
                    {...methods.getInputProps("coresetSize")}
                />

                <NumberInput
                    label="Inference Threshold"
                    description="Converts a raw score to a binary decision with threshold"
                    placeholder="-470.8"
                    mt="xs"
                    {...methods.getInputProps("threshold")}
                />
                <Switch
                    color="green"
                    label="Training mode"
                    description="All logs are assumed to be normal"
                    size="md"
                    mt="lg"
                    mb="md"
                    {...methods.getInputProps("isTrain")}
                />
                <div className="flex justify-end pt-4">
                    <Button type="submit" radius="md">
                        Add Service
                    </Button>
                </div>
            </form>
        </Modal>
    );
};
