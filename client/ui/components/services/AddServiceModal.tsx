import { useForm } from "@mantine/form";
import { zodResolver } from "mantine-form-zod-resolver";
import { useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { trpc } from "../../../utils/trpc";
import { Button, Modal, NumberInput, Switch, TextInput } from "@mantine/core";

const createServiceSchema = z.object({
    name: z.string(),
    description: z.string(),
    isTrain: z.boolean().default(false),
    threshold: z.number().optional(),
    coresetSize: z.number().optional(),
});

type CreateServiceInput = z.TypeOf<typeof createServiceSchema>;

interface Props {
    // State of the modal
    opened: boolean;
    open: () => void;
    close: () => void;
}

export const AddService = ({ opened, open, close }: Props) => {
    const utils = trpc.useUtils();

    const { mutate: createService } = trpc.services.createService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
            close();
        },
    });

    const methods = useForm<CreateServiceInput>({
        validate: zodResolver(createServiceSchema),
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
