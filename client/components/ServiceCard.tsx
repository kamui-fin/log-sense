import { IconHeart, IconTrash } from "@tabler/icons-react";
import { useForm } from "@mantine/form";
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

interface Service {
  _id: string;
  name: string;
  description: string;
  isTrain: string;
  threshold: number;
  coresetSize: number;
}

interface ServiceCardProps {
  service: Service;
}

export function ServiceCard({ service }: ServiceCardProps) {
  const { name, description, isTrain: mode, threshold, coresetSize: coresetNumber } = service;
  // update hyperparameters form
  const form = useForm({
    initialValues: {
      mode: "train",
      threshold: -470.0,
      coresetSize: 2,
    },
  });

  return (
    <Card withBorder radius="md" p="md" className={classes.card}>
      <form onSubmit={form.onSubmit((values) => console.log(values))}>
        <Card.Section className={classes.section} mt="sm">
          <Group justify="apart">
            <Text fz="lg" fw={500}>
              {name}
            </Text>
          </Group>
          <Text fz="sm" mt="xs">
            {description}
          </Text>
          <Select
            label="Mode"
            description="In train mode, all logs are recognized as normal"
            placeholder="Pick value"
            data={["Train", "Inference"]}
            defaultValue="Inference"
            allowDeselect={false}
            {...form.getInputProps("mode")}
          />

          <NumberInput
            label="Coreset Size"
            description="Number of neighbors to use for RAPID"
            placeholder="2"
            {...form.getInputProps("coresetSize")}
          />

          <NumberInput
            label="Inference Threshold"
            description="Converts a raw score to a binary decision with threshold"
            placeholder="-470.8"
            {...form.getInputProps("threshold")}
          />
        </Card.Section>

        <Group mt="xs">
          <Button type="submit" radius="md" style={{ flex: 1 }}>
            Save Config
          </Button>
          <ActionIcon variant="default" radius="md" size={36}>
            <IconTrash className={classes.like} stroke={1.5} />
          </ActionIcon>
        </Group>
      </form>
    </Card>
  );
}
