import { useForm, zodResolver } from "@mantine/form";
import { useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { trpc } from "../../../utils/trpc";
import {
    Button,
    Modal,
    NumberInput,
    Switch,
    TextInput,
    Textarea,
} from "@mantine/core";

const createServiceSchema = z.object({
    name: z.string(),
    description: z.string(),
    threshold: z.number().optional(),
    is_train: z.boolean().default(false),
    coreset_size: z.number().default(2),
    enable_trace: z.boolean().default(false),
    trace_regex: z.string().optional(),
    top_k: z.number().default(80),
    max_pretrain: z.number().default(10_000),
    lr_pretraining: z.number().default(1e-4),
    lr_finetuning: z.number().default(1e-6),
    train_batch_size: z.number().default(16),
    num_episodes: z.number().default(10),
    num_epochs: z.number().default(10),
    vocab_size: z.number().default(500),
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
                <Textarea
                    label="Description"
                    fz="sm"
                    mt="xs"
                    {...methods.getInputProps("description")}
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
