import { NumberInput, Button } from "@mantine/core";
import { ServiceCardProps } from "../ServiceCard";
import { useForm, zodResolver } from "@mantine/form";
import { z } from "zod";
import { FormEventHandler } from "react";
import { trpc } from "../../../../utils/trpc";


type UpdateRAPIDInput = z.TypeOf<typeof updateRAPIDSchema>;

const updateRAPIDSchema = z.object({
    threshold: z.number().optional(),
    coreset_size: z.number().optional(),
});

export const RAPIDTab = ({ service, onGoBack }: ServiceCardProps) => {
    const { coreset_size, threshold } = service;

    const form = useForm<UpdateRAPIDInput>({
        initialValues: {
            coreset_size,
            threshold,
        },
        validate: zodResolver(updateRAPIDSchema),
    });

    const utils = trpc.useUtils();
    const { mutate: updateService } = trpc.services.updateService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });

    const submitRAPID: FormEventHandler = (values) => {
        updateService({
            params: { id: service._id, type: "SERVICE" },
            body: values,
        });
        onGoBack();
    };

    return (
        <div className="grid grid-cols-3 gap-6">
            <form onSubmit={form.onSubmit((values) => submitRAPID(values))}>
                <NumberInput
                    label="Coreset Size"
                    description="Number of neighbors to use for RAPID"
                    placeholder="2"
                    mt="md"
                    {...form.getInputProps("coreset_size")}
                />
                <NumberInput
                    label="Inference Threshold"
                    description="Converts a raw score to a binary decision with threshold"
                    placeholder="-470.8"
                    mt="xs"
                    {...form.getInputProps("threshold")}
                />
                <div className="flex justify-start pt-4">
                    <Button type="submit" radius="md">
                        Update Config
                    </Button>
                </div>
            </form>
        </div>
    );
};
