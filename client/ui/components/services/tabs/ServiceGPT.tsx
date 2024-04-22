import { NumberInput, Switch, Button } from "@mantine/core";
import z from "zod";
import { ServiceCardProps } from "../ServiceCard";
import { useForm, zodResolver } from "@mantine/form";
import { FormEventHandler } from "react";
import { trpc } from "../../../../utils/trpc";

/*
    top-k -- number
    max pretrain -- number
    vocab size -- number

    lr pretraining -- number
    lr finetuning -- number
    train batch size -- number
    num episodes -- number
    num epochs -- number

    enable trace -- switch
*/

type UpdateGPTInput = z.TypeOf<typeof updateGPTSchema>;

const updateGPTSchema = z.object({
    top_k: z.number().optional(),
    max_pretrain: z.number().optional(),
    vocab_size: z.number().optional(),
    lr_pretraining: z.number().optional(),
    lr_finetuning: z.number().optional(),
    train_batch_size: z.number().optional(),
    num_episodes: z.number().optional(),
    num_epochs: z.number().optional(),
    enable_trace: z.boolean().default(false),
});

export const GPTTab = ({ service, onGoBack }: ServiceCardProps) => {
    const {
        top_k,
        max_pretrain,
        vocab_size,
        lr_pretraining,
        lr_finetuning,
        train_batch_size,
        num_episodes,
        num_epochs,
        enable_trace,
    } = service;

    const form = useForm<UpdateGPTInput>({
        initialValues: {
            top_k,
            max_pretrain,
            vocab_size,
            lr_pretraining,
            lr_finetuning,
            train_batch_size,
            num_episodes,
            num_epochs,
            enable_trace,
        },
        validate: zodResolver(updateGPTSchema),
    });

    const utils = trpc.useUtils();
    const { mutate: updateService } = trpc.services.updateService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });

    const submitGPT = (values: UpdateGPTInput) => {
        updateService({
            params: { id: service._id },
            body: values,
        });
    };

    return (
        <div className="mb-4">
            <form
                onSubmit={form.onSubmit((values) => submitGPT(values))}
                className="grid content-center justify-center grid-cols-4 gap-10 mb-4 w-full"
            >
                <NumberInput
                    label="Top-K"
                    description="K-value during Top-K evaluation"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("top_k")}
                />
                <NumberInput
                    label="Number of Pretraining Logs"
                    description="# of log lines to pretrain before switching to finetuning"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("max_pretrain")}
                />
                <NumberInput
                    label="Vocab Size"
                    description="Number of unique, cleaned log lines to expect"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("vocab_size")}
                />

                <NumberInput
                    label="Pretraining LR"
                    description="Learning Rate for Pretraining"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("lr_pretraining")}
                />
                <NumberInput
                    label="Finetuning LR"
                    description="Learning Rate for Finetuning"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("lr_finetuning")}
                />
                <NumberInput
                    label="Train Batch size"
                    description="Number of log lines sent to GPU per batch"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("train_batch_size")}
                />
                <NumberInput
                    label="Episodes"
                    description="Number of Episodes using during Pretraining and Finetuning"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("num_episodes")}
                />
                <NumberInput
                    label="Epochs"
                    description="Number of Epochs using during Pretraining and Finetuning"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("num_epochs")}
                />
                <Switch
                    color="green"
                    label="Enable Trace"
                    size="md"
                    mt="lg"
                    mb="md"
                    defaultChecked={enable_trace}
                    {...form.getInputProps("enable_trace")}
                />
            </form>
            <div className="flex justify-start pt-4">
                <Button type="submit" radius="md">
                    Update Config
                </Button>
            </div>
        </div>
    );
};
