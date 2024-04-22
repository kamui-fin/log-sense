import { NumberInput, Button } from "@mantine/core"
import { ServiceCardProps } from "../ServiceCard";
import { useForm, zodResolver } from "@mantine/form";
import { z } from "zod";


type UpdateRAPIDInput = z.TypeOf<typeof updateRAPIDSchema>;

const updateRAPIDSchema = z.object({
    threshold: z.number().optional(),
    coreset_size: z.number().optional(),
});

export const RAPIDTab = ({service, onSubmitTab}: ServiceCardProps) => {
    const {coreset_size, threshold} = service;

    const form = useForm<UpdateRAPIDInput>({
        initialValues: {
            coreset_size,
            threshold,
            resolver: zodResolver(updateRAPIDSchema),
        },
    });

    const submitToTop = (e) => {
        e.preventDefault();
        console.log(service)
        onSubmitTab(form.getValues())
    }

    return(
        <div className="grid grid-cols-3 gap-6">
            <form onSubmit={submitToTop}>
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
        
    )
}