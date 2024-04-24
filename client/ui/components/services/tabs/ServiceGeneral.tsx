import { TextInput, Switch, Textarea } from "@mantine/core";
import { useState } from "react";
import { Tooltip, Center, Text, rem } from "@mantine/core";
import { IconAlertTriangle} from "@tabler/icons-react";
import { useForm, zodResolver } from "@mantine/form";
import { z } from "zod";
import { trpc } from "../../../../utils/trpc";
import { Button } from "@mantine/core";
import classes from './InputValidation.module.css';
import { Service } from "../ServiceCard";

type UpdateGeneralInput = z.TypeOf<typeof updateGeneralSchema>;

const updateGeneralSchema = z.object({
    name: z.string(),
    description: z.string(),
    is_train: z.boolean(),
});

export interface GenTabProps {
    service: Service;
    onChangeName(name: string): void;
}

export const GeneralTab = ({ service, onChangeName }: GenTabProps) => {
    // const {name, description, is_train} = service
    const [nameValid, setNameValid] = useState(true)

    const [genService, setGenService] = useState({name: service.name, description: service.description, is_train: service.is_train})
    const [descriptionValid, setDescriptionValid] = useState(true)

    const form = useForm<UpdateGeneralInput>({
        initialValues: {
            name:genService.name,
            description:genService.description,
            is_train:genService.is_train,
        },
        validate: zodResolver(updateGeneralSchema),
    });

    const utils = trpc.useUtils();
    const { mutate: updateService } = trpc.services.updateService.useMutation({
        onSuccess() {
            utils.services.getServices.invalidate();
        },
    });

    const submitGeneral = (e: React.MouseEvent<HTMLButtonElement>) => {
        e.preventDefault()
        if(genService.name.length == 0){
            setNameValid(false);
        }
        else if(genService.description.length > 450){
            setDescriptionValid(false)
        }
        else{
            setNameValid(true)
            setDescriptionValid(true)
            onChangeName(genService.name)
            updateService({
                params: { id: String(service._id) },
                body: genService,
            });
        }
    };

    const getRight = (warning: String) => {
        return (
                <Tooltip
                label={warning}
                position="top-end"
                withArrow
                transitionProps={{ transition: 'pop-bottom-right' }}
                >
                <Text component="div" c="dimmed" style={{ cursor: 'help' }}>
                    <Center>
                        <IconAlertTriangle
                            stroke={1.5}
                            style={{ width: rem(18), height: rem(18) }}
                            className={classes.icon}
                        />
                    </Center>
                </Text>
                </Tooltip>
            );
    }

    const getNameRight = (warning: String) => {
        if(nameValid) return;
        return getRight(warning);
    }

    const getDescriptionRight = (warning: String) => {
        if(descriptionValid) return;
        return getRight(warning);
    }

    return (
        <div>
            <div className="grid grid-cols-3 gap-10 mb-4">
                <div>
                <TextInput
                        label="Name"
                        description=''
                        mt="md"
                        fz="lg"
                        fw={500}
                        onChange={(e) => setGenService({...genService, name: e.target.value.trim()})}
                        classNames={{ input: ((nameValid) ? classes.valid : classes.invalid) }}
                        defaultValue={genService.name}
                        rightSection={getNameRight('Name can\'t be empty')}
                    />
                    <Textarea
                        label="Description"
                        description={<div>You have {(450 < genService.description.length) ? 0 : 450 - genService.description.length} chars left</div>}
                        fz="sm"
                        mt="xs"
                        autosize
                        onChange={((e) => setGenService({...genService, description: e.target.value.trim()}))}
                        classNames={{ input: ((descriptionValid) ? classes.valid : classes.invalid) }}
                        rightSection={getDescriptionRight('Description is too long.')}
                        defaultValue={genService.description}
                    />
                    <Switch
                        color="green"
                        label="Training mode"
                        description="All logs are assumed to be normal"
                        size="md"
                        mt="lg"
                        mb="xs"
                        defaultChecked={genService.is_train}
                        onChange={(e) => setGenService({...genService, is_train: e.target.checked})}
                    />
                    <div className="flex justify-start pt-4">
                        <Button onClick={(e) => submitGeneral(e)} radius="md">
                            Update Config
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
};
