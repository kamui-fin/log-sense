import { IconHeart, IconTrash } from "@tabler/icons-react";
import { useForm, zodResolver } from "@mantine/form";
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
} from "@mantine/core";
import classes from "./ServiceCard.module.css";
import { Switch } from "@mantine/core";
import { useQueryClient } from "@tanstack/react-query";
import { trpc } from "../../../utils/trpc";
import { z } from "zod";
import { Update } from "next/dist/build/swc";
import { ObjectId } from "mongoose";

interface Service {
  _id: string;
  name: string;
  description: string;
  isTrain: boolean;
  threshold: number;
  coresetSize: number;
}

interface ServiceCardProps {
  service: Service;
}

const updateServiceSchema = z.object({
  isTrain: z.boolean().default(false),
  threshold: z.number().optional(),
  coresetSize: z.number().optional(),
});

type UpdateServiceInput = z.TypeOf<typeof updateServiceSchema>;

export function ServiceCard({ service }: ServiceCardProps) {
  const { name, description, isTrain, threshold, coresetSize } = service;
  const form = useForm<UpdateServiceInput>({
    initialValues: {
      isTrain,
      threshold,
      coresetSize,
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
    <Card withBorder radius="md" p="md" className={classes.card}>
      <form onSubmit={onUpdateHandler}>
        <Card.Section className={classes.section} mt="sm">
          <Text fz="lg" fw={700}>
            {name}
          </Text>
          <Text fz="sm" mt="xs" c="dimmed">
            {description}
          </Text>

          <NumberInput
            label="Coreset Size"
            description="Number of neighbors to use for RAPID"
            placeholder="2"
            mt="md"
            {...form.getInputProps("coresetSize")}
          />

          <NumberInput
            label="Inference Threshold"
            description="Converts a raw score to a binary decision with threshold"
            placeholder="-470.8"
            mt="xs"
            {...form.getInputProps("threshold")}
          />

          <Switch
            color="green"
            label="Training mode"
            description="All logs are assumed to be normal"
            size="md"
            mt="lg"
            mb="xs"
            defaultChecked={isTrain}
            {...form.getInputProps("isTrain")}
          />
        </Card.Section>

        <Group mt="xs">
          <Button type="submit" radius="md" style={{ flex: 1 }}>
            Save Config
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
      </form>
    </Card>
  );
}
